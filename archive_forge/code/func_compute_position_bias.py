from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mpnet import MPNetConfig
def compute_position_bias(self, x, position_ids=None):
    """Compute binned relative position bias"""
    input_shape = shape_list(x)
    qlen, klen = (input_shape[1], input_shape[1])
    if position_ids is not None:
        context_position = position_ids[:, :, None]
        memory_position = position_ids[:, None, :]
    else:
        context_position = tf.range(qlen)[:, None]
        memory_position = tf.range(klen)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = self._relative_position_bucket(relative_position, num_buckets=self.relative_attention_num_buckets)
    values = tf.gather(self.relative_attention_bias, rp_bucket)
    values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
    return values