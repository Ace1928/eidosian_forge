from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
def get_attn(self, attn: tf.Tensor, gumbel: bool=True, hard: bool=True, training: bool=False) -> tf.Tensor:
    if gumbel and training:
        attn = gumbel_softmax(attn, dim=-2, hard=hard)
    elif hard:
        attn = hard_softmax(attn, dim=-2)
    else:
        attn = stable_softmax(attn, axis=-2)
    return attn