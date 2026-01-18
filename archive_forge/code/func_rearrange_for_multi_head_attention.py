from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
    batch_size, hidden_size, _ = shape_list(hidden_state)
    head_dim = self.embed_dim // self.num_heads
    hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
    hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
    return hidden_state