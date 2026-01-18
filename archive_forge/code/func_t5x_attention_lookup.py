import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def t5x_attention_lookup(params, i, prefix, layer_name='attention'):
    """Returns the KOQV parameters of (self-)attention. Does not transpose."""
    k_tmp = k_tmp = np.ascontiguousarray(params[f'{prefix}/{prefix}/{layer_name}/key/kernel'][:, i, :, :])
    k = k_tmp.reshape(k_tmp.shape[0], k_tmp.shape[1] * k_tmp.shape[2])
    o_tmp = np.ascontiguousarray(params[f'{prefix}/{prefix}/{layer_name}/out/kernel'][:, i, :, :])
    o = o_tmp.reshape(o_tmp.shape[0] * o_tmp.shape[1], o_tmp.shape[2])
    q_tmp = np.ascontiguousarray(params[f'{prefix}/{prefix}/{layer_name}/query/kernel'][:, i, :, :])
    q = q_tmp.reshape(q_tmp.shape[0], q_tmp.shape[1] * q_tmp.shape[2])
    v_tmp = np.ascontiguousarray(params[f'{prefix}/{prefix}/{layer_name}/value/kernel'][:, i, :, :])
    v = v_tmp.reshape(v_tmp.shape[0], v_tmp.shape[1] * v_tmp.shape[2])
    return (k, o, q, v)