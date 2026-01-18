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
def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor:
    y_soft = stable_softmax(logits, dim)
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(index, depth=shape_list(logits)[dim], axis=range(len(shape_list(logits)))[dim], dtype=y_soft.dtype)
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    return ret