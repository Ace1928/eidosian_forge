import argparse
import tensorflow as tf
import torch
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging
def get_encoder_attention_layer_array(layer_index: int, name: str, orginal_shape):
    full_name = f'encoder/_transformer_layers/{layer_index}/_attention_layer/{name}/.ATTRIBUTES/VARIABLE_VALUE'
    array = tf.train.load_variable(tf_checkpoint_path, full_name)
    array = array.reshape(orginal_shape)
    if 'kernel' in name:
        array = array.transpose()
    return torch.from_numpy(array)