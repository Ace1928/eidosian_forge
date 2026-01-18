from __future__ import annotations
import math
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig
def embed_positions(self, position_ids: np.ndarray | tf.Tensor | None=None) -> tf.Tensor:
    position_ids += self.offset
    positions = tf.gather(self._embed_positions_weights, position_ids, axis=0)
    return positions