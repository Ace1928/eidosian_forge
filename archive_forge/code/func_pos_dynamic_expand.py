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
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [shape_list(pos_index)[-2], shape_list(key_layer)[-2]]
    return tf.broadcast_to(pos_index, shapes)