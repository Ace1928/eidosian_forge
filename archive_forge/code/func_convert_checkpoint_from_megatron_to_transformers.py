import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ['mp_rank_00', 'mp_rank_00_000']
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f'Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}')
    state_dict = torch.load(rank0_checkpoint_path, map_location='cpu')
    megatron_args = state_dict.get('args', None)
    if megatron_args is None:
        raise ValueError('Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints containing all the megatron arguments. This is because it loads all config related to model architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron arguments to use this utility.')
    if megatron_args is not None:
        if megatron_args.bias_gelu_fusion:
            activation_function = 'gelu_fast'
        elif megatron_args.openai_gelu:
            activation_function = 'gelu_new'
        else:
            activation_function = 'gelu'
    else:
        activation_function = 'gelu_new'
    vocab_size = megatron_args.padded_vocab_size if getattr(megatron_args, 'orig_vocab_size', None) is None else megatron_args.orig_vocab_size
    print(vocab_size)
    config = GPT2Config(vocab_size=vocab_size, n_positions=megatron_args.max_position_embeddings, n_embd=megatron_args.hidden_size, n_layer=megatron_args.num_layers, n_head=megatron_args.num_attention_heads, n_inner=megatron_args.ffn_hidden_size, activation_function=activation_function, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, scale_attn_weights=True, use_cache=True, bos_token_id=vocab_size - 1, eos_token_id=vocab_size - 1, architectures=['GPT2LMHeadModel'])
    output_state_dict = {}
    checkpoint_version = state_dict.get('checkpoint_version', 0.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.float32
    layer_re = re.compile('layers\\.(\\d+)\\.([a-z0-9_.]+)\\.([a-z]+)')
    print('Converting')
    print('Converting embeddings')
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    position_embeddings = get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model.embedding.position_embeddings.weight')
    output_state_dict['transformer.wpe.weight'] = position_embeddings.to(dtype)
    word_embeddings = torch.cat([get_element_from_dict_by_path(tp_state_dicts[tp_rank], 'model.language_model.embedding.word_embeddings.weight') for tp_rank in range(tp_size)], dim=0)
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict['transformer.wte.weight'] = word_embeddings
    print('Converting transformer layers')
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head
    n_positions = config.n_positions
    num_layers = config.num_hidden_layers // pp_size
    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f'Converting pipeline parallel rank {pp_rank}')
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)
        path = 'model.language_model.transformer' if 'transformer' in get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model').keys() else 'model.language_model.encoder'
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            m = layer_re.match(key)
            if m is None:
                break
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            layer_name = f'transformer.h.{layer_idx}'
            if op_name + '.' + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ['self_attention.dense', 'mlp.dense_4h_to_h', 'attention.dense'] else 0
                params = torch.cat([val] + [get_element_from_dict_by_path(tp_state_dicts[tp_rank], f'{path}')[key] for tp_rank in range(1, tp_size)], dim=dim).to(dtype)
            if op_name.endswith('layernorm'):
                ln_name = 'ln_1' if op_name.startswith('input') else 'ln_2'
                output_state_dict[layer_name + '.' + ln_name + '.' + weight_or_bias] = params
            elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'weight':
                causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=dtype)).view(1, 1, n_positions, n_positions)
                output_state_dict[layer_name + '.attn.bias'] = causal_mask
                masked_bias = torch.tensor(-10000.0, dtype=dtype)
                output_state_dict[layer_name + '.attn.masked_bias'] = masked_bias
                out_val = megatron_to_transformers_fix_query_key_value_ordering(params, checkpoint_version, 3, heads, hidden_size_per_head)
                out_val = out_val.transpose(0, 1).contiguous()
                output_state_dict[layer_name + '.attn.c_attn.weight'] = out_val
            elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'bias':
                out_val = megatron_to_transformers_fix_query_key_value_ordering(params, checkpoint_version, 3, heads, hidden_size_per_head)
                output_state_dict[layer_name + '.attn.c_attn.bias'] = out_val
            elif weight_or_bias == 'weight':
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + 'weight'] = params.transpose(0, 1)
            elif weight_or_bias == 'bias':
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + 'bias'] = params
    if config.n_layer != layer_idx + 1:
        raise ValueError(f'Expected {config.n_layer} layers but found {layer_idx + 1}')
    print('Converting final layernorm')
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict['transformer.ln_f.weight'] = params['final_layernorm.weight'].to(dtype)
    output_state_dict['transformer.ln_f.bias'] = params['final_layernorm.bias'].to(dtype)
    print('Converting LM head')
    output_state_dict['lm_head.weight'] = word_embeddings.to(dtype)
    print('Conversion from Megatron-LM to Transformers is done!')
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    if args.tokenizer_name is None:
        tokenizer_name = 'openai-community/gpt2'
    else:
        tokenizer_name = args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class
    print('Saving config')
    config.save_pretrained(args.save_path)
    if args.tokenizer_name is not None:
        print(f'Adding {tokenizer_class} tokenizer files')
        tokenizer.save_pretrained(args.save_path)
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))
    if index is None:
        print(f'Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}')
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        with open(save_index_file, 'w', encoding='utf-8') as f:
            content = json.dumps(index, indent=2, sort_keys=True) + '\n'
            f.write(content)
        print(f'The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the index located at {save_index_file}.')