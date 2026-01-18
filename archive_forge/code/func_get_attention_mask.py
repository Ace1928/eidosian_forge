from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
def get_attention_mask(self, attention_mask):
    if len(shape_list(attention_mask)) <= 2:
        extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
        attention_mask = extended_attention_mask * tf.expand_dims(tf.squeeze(extended_attention_mask, -2), -1)
        attention_mask = tf.cast(attention_mask, tf.uint8)
    elif len(shape_list(attention_mask)) == 3:
        attention_mask = tf.expand_dims(attention_mask, 1)
    return attention_mask