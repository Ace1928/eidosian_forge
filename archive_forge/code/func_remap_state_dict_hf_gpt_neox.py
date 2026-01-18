import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, GPTNeoXConfig
def remap_state_dict_hf_gpt_neox(state_dict, config):

    def key_mapping_layers(key):
        return re.sub('^gpt_neox.', 'transformer.', key)
    state_dict = OrderedDict(((key_mapping_layers(k), v) for k, v in state_dict.items()))

    def key_mapping_emb(key):
        return re.sub('^transformer.embed_in.', 'transformer.embeddings.word_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, 'tie_word_embeddings', False):
        state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']
    else:
        output_embeddings = state_dict.pop('embed_out.weight')
        state_dict['lm_head.weight'] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))

    def key_mapping_ln(key):
        key = re.sub('^transformer.final_layer_norm.', 'transformer.ln_f.', key)
        key = re.sub('^transformer.layers.(\\d+).input_layernorm.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^transformer.layers.(\\d+).post_attention_layernorm.', 'transformer.layers.\\1.norm2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_h_to_4h.', 'transformer.layers.\\1.mlp.fc1.', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_4h_to_h.', 'transformer.layers.\\1.mlp.fc2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for l in range(config.n_layer):
        state_dict.pop(f'transformer.layers.{l}.attention.bias')
        state_dict.pop(f'transformer.layers.{l}.attention.masked_bias')
        state_dict.pop(f'transformer.layers.{l}.attention.rotary_emb.inv_freq', None)
        headdim = config.hidden_size // config.num_attention_heads
        Wqkv = state_dict.pop(f'transformer.layers.{l}.attention.query_key_value.weight')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = rearrange(Wqkv, '(nheads three headdim) ... -> (three nheads headdim) ...', three=3, headdim=headdim)
        bqkv = state_dict.pop(f'transformer.layers.{l}.attention.query_key_value.bias')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.bias'] = rearrange(bqkv, '(nheads three headdim) -> (three nheads headdim)', three=3, headdim=headdim)

    def key_mapping_attn(key):
        key = re.sub('^transformer.layers.(\\d+).attention.dense.', 'transformer.layers.\\1.mixer.out_proj.', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict