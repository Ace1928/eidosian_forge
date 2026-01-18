from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def get_parlai_opt(self) -> Opt:
    """
        Parser for converting fairseq argument to ParlAI opt.

        :return opt:
            opt parsed by ParlAI Parser
        """
    state = self.state
    fairseq_args = state['args'].__dict__
    transformer_common_config = {}
    for each in TRANSFORMER_PARAMETER_MAPPING:
        transformer_common_config[TRANSFORMER_PARAMETER_MAPPING[each]] = fairseq_args[f'encoder_{each}']
    for each in TRANSFORMER_DROPOUT:
        transformer_common_config[each] = fairseq_args[each]
    if 'activation_dropout' in fairseq_args:
        transformer_common_config['relu_dropout'] = fairseq_args['activation_dropout']
    else:
        transformer_common_config['relu_dropout'] = fairseq_args['relu_dropout']
    transformer_common_config.update({'model': self.opt['model'], 'n_encoder_layers': fairseq_args['encoder_layers'], 'n_decoder_layers': fairseq_args['decoder_layers'], 'dict_tokenizer': self.opt['tokenizer'], 'bpe_vocab': self.opt['vocab'], 'bpe_merge': self.opt['merge'], 'n_positions': fairseq_args['max_source_positions']})
    if 'encoder_embed_scale' in fairseq_args:
        transformer_common_config['embeddings_scale'] = fairseq_args['encoder_embed_scale'] != 1.0
    else:
        transformer_common_config['embeddings_scale'] = not fairseq_args['no_scale_embedding']
    if fairseq_args['encoder_normalize_before']:
        transformer_common_config['variant'] = 'prelayernorm'
    elif fairseq_args['layernorm_embedding']:
        transformer_common_config['variant'] = 'bart'
    else:
        transformer_common_config['variant'] = 'aiayn'
    if self.opt['add_prefix_space']:
        transformer_common_config['bpe_add_prefix_space'] = True
    parser = ParlaiParser()
    parser.set_params(**transformer_common_config)
    opt = parser.parse_args([])
    opt['fp16'] = self.opt['fp16']
    opt['activation'] = self.opt['activation']
    opt['delimiter'] = self.opt['delimiter']
    opt['history_add_global_end_token'] = self.opt['history_add_global_end_token']
    opt['force_fp16_tokens'] = True
    opt['converting'] = True
    return opt