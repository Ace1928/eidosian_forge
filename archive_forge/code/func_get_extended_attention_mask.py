from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_electra import ElectraConfig
def get_extended_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length=0):
    batch_size, seq_length = input_shape
    if attention_mask is None:
        attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)
    attention_mask_shape = shape_list(attention_mask)
    mask_seq_length = seq_length + past_key_values_length
    if self.is_decoder:
        seq_ids = tf.range(mask_seq_length)
        causal_mask = tf.less_equal(tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None])
        causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
        extended_attention_mask = causal_mask * attention_mask[:, None, :]
        attention_mask_shape = shape_list(extended_attention_mask)
        extended_attention_mask = tf.reshape(extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2]))
        if past_key_values_length > 0:
            extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
    else:
        extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
    extended_attention_mask = tf.cast(extended_attention_mask, dtype=dtype)
    one_cst = tf.constant(1.0, dtype=dtype)
    ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
    extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
    return extended_attention_mask