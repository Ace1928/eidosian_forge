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
def take_along_axis(x, indices):
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
        one_hot_indices = tf.one_hot(indices, depth=x.shape[-1], dtype=x.dtype)
        gathered = tf.einsum('ijkl,ijl->ijk', one_hot_indices, x)
    else:
        gathered = tf.gather(x, indices, batch_dims=2)
    return gathered