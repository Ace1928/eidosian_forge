import argparse
import json
import os
import re
import shutil
import torch
from transformers import BioGptConfig, BioGptForCausalLM
from transformers.models.biogpt.tokenization_biogpt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging
def convert_biogpt_checkpoint_to_pytorch(biogpt_checkpoint_path, pytorch_dump_folder_path):
    if not os.path.exists(biogpt_checkpoint_path):
        raise ValueError(f'path {biogpt_checkpoint_path} does not exist!')
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f'Writing results to {pytorch_dump_folder_path}')
    checkpoint_file = os.path.join(biogpt_checkpoint_path, 'checkpoint.pt')
    if not os.path.isfile(checkpoint_file):
        raise ValueError(f'path to the file {checkpoint_file} does not exist!')
    chkpt = torch.load(checkpoint_file, map_location='cpu')
    args = chkpt['cfg']['model']
    dict_file = os.path.join(biogpt_checkpoint_path, 'dict.txt')
    if not os.path.isfile(dict_file):
        raise ValueError(f'path to the file {dict_file} does not exist!')
    src_dict = Dictionary.load(dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES['vocab_file'])
    print(f'Generating {src_vocab_file} of {src_vocab_size} records')
    with open(src_vocab_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))
    bpecodes_file = os.path.join(biogpt_checkpoint_path, 'bpecodes')
    if not os.path.isfile(bpecodes_file):
        raise ValueError(f'path to the file {bpecodes_file} does not exist!')
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES['merges_file'])
    shutil.copyfile(bpecodes_file, merges_file)
    biogpt_model_config_file = os.path.join(pytorch_dump_folder_path, 'config.json')
    model_conf = {'activation_dropout': args['activation_dropout'], 'architectures': ['BioGptForCausalLM'], 'attention_probs_dropout_prob': args['attention_dropout'], 'bos_token_id': 0, 'eos_token_id': 2, 'hidden_act': args['activation_fn'], 'hidden_dropout_prob': args['dropout'], 'hidden_size': args['decoder_embed_dim'], 'initializer_range': 0.02, 'intermediate_size': args['decoder_ffn_embed_dim'], 'layer_norm_eps': 1e-12, 'layerdrop': args['decoder_layerdrop'], 'max_position_embeddings': args['max_target_positions'], 'model_type': 'biogpt', 'num_attention_heads': args['decoder_attention_heads'], 'num_hidden_layers': args['decoder_layers'], 'pad_token_id': 1, 'scale_embedding': not args['no_scale_embedding'], 'tie_word_embeddings': args['share_decoder_input_output_embed'], 'vocab_size': src_vocab_size}
    print(f'Generating {biogpt_model_config_file}')
    with open(biogpt_model_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))
    biogpt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)
    tokenizer_conf = {'bos_token': '<s>', 'eos_token': '</s>', 'model_max_length': 1024, 'pad_token': '<pad>', 'special_tokens_map_file': None, 'tokenizer_class': 'BioGptTokenizer', 'unk_token': '<unk>'}
    print(f'Generating {biogpt_tokenizer_config_file}')
    with open(biogpt_tokenizer_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))
    model_state_dict = chkpt['model']
    ignore_keys = ['decoder.version']
    for k in ignore_keys:
        model_state_dict.pop(k, None)
    layer_names = list(model_state_dict.keys())
    for layer_name in layer_names:
        if layer_name.endswith('output_projection.weight'):
            model_state_dict[layer_name.replace('decoder.', '')] = model_state_dict.pop(layer_name)
        else:
            model_state_dict[layer_name.replace('decoder', 'biogpt')] = model_state_dict.pop(layer_name)
    config = BioGptConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = BioGptForCausalLM(config)
    model_new.load_state_dict(model_state_dict)
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f'Generating {pytorch_weights_dump_path}')
    torch.save(model_state_dict, pytorch_weights_dump_path)
    print('Conversion is done!')