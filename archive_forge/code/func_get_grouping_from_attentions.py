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
def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: Tuple[int]) -> tf.Tensor:
    """
    Args:
        attentions (`tuple(tf.Tensor)`: tuple of attention maps returned by `TFGroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `tf.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """
    attn_maps = []
    prev_attn_masks = None
    for attn_masks in attentions:
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = tf.matmul(prev_attn_masks, attn_masks)
        cur_attn_map = resize_attention_map(tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape)
        attn_maps.append(cur_attn_map)
    final_grouping = attn_maps[-1]
    return tf.stop_gradient(final_grouping)