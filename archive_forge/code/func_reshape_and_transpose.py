from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
def reshape_and_transpose(self, vector, batch_size):
    return tf.reshape(tf.transpose(tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)), (0, 2, 1, 3)), (batch_size * self.num_heads, -1, self.head_dim))