from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
def post_attention(self, h, attn_vec, residual=True, training=False):
    """Post-attention processing."""
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
    attn_out = self.dropout(attn_out, training=training)
    if residual:
        attn_out = attn_out + h
    output = self.layer_norm(attn_out)
    return output