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
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [shape_list(query_layer)[0], shape_list(query_layer)[1], shape_list(query_layer)[2], shape_list(relative_pos)[-1]]
    return tf.broadcast_to(c2p_pos, shapes)