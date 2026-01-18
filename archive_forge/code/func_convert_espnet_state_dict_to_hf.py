import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import yaml
from transformers import (
def convert_espnet_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if 'tts.generator.text2mel.' in key:
            new_key = key.replace('tts.generator.text2mel.', '')
            if 'postnet' in key:
                new_key = new_key.replace('postnet.postnet', 'speech_decoder_postnet.layers')
                new_key = new_key.replace('.0.weight', '.conv.weight')
                new_key = new_key.replace('.1.weight', '.batch_norm.weight')
                new_key = new_key.replace('.1.bias', '.batch_norm.bias')
                new_key = new_key.replace('.1.running_mean', '.batch_norm.running_mean')
                new_key = new_key.replace('.1.running_var', '.batch_norm.running_var')
                new_key = new_key.replace('.1.num_batches_tracked', '.batch_norm.num_batches_tracked')
            if 'feat_out' in key:
                if 'weight' in key:
                    new_key = 'speech_decoder_postnet.feat_out.weight'
                if 'bias' in key:
                    new_key = 'speech_decoder_postnet.feat_out.bias'
            if 'encoder.embed.0.weight' in key:
                new_key = new_key.replace('0.', '')
            if 'w_1' in key:
                new_key = new_key.replace('w_1', 'conv1')
            if 'w_2' in key:
                new_key = new_key.replace('w_2', 'conv2')
            if 'predictor.conv' in key:
                new_key = new_key.replace('.conv', '.conv_layers')
                pattern = '(\\d)\\.(\\d)'
                replacement = '\\1.conv' if '2.weight' not in new_key and '2.bias' not in new_key else '\\1.layer_norm'
                new_key = re.sub(pattern, replacement, new_key)
            if 'pitch_embed' in key or 'energy_embed' in key:
                new_key = new_key.replace('0', 'conv')
            if 'encoders' in key:
                new_key = new_key.replace('encoders', 'conformer_layers')
                new_key = new_key.replace('norm_final', 'final_layer_norm')
                new_key = new_key.replace('norm_mha', 'self_attn_layer_norm')
                new_key = new_key.replace('norm_ff_macaron', 'ff_macaron_layer_norm')
                new_key = new_key.replace('norm_ff', 'ff_layer_norm')
                new_key = new_key.replace('norm_conv', 'conv_layer_norm')
            if 'lid_emb' in key:
                new_key = new_key.replace('lid_emb', 'language_id_embedding')
            if 'sid_emb' in key:
                new_key = new_key.replace('sid_emb', 'speaker_id_embedding')
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict