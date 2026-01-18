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
def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
    if relative_pos is None:
        q = shape_list(query_layer)[-2]
        relative_pos = build_relative_position(q, shape_list(key_layer)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
    shape_list_pos = shape_list(relative_pos)
    if len(shape_list_pos) == 2:
        relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
    elif len(shape_list_pos) == 3:
        relative_pos = tf.expand_dims(relative_pos, 1)
    elif len(shape_list_pos) != 4:
        raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}')
    att_span = self.pos_ebd_size
    rel_embeddings = tf.expand_dims(rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :], 0)
    if self.share_att_key:
        pos_query_layer = tf.tile(self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
        pos_key_layer = tf.tile(self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
    else:
        if 'c2p' in self.pos_att_type:
            pos_key_layer = tf.tile(self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
        if 'p2c' in self.pos_att_type:
            pos_query_layer = tf.tile(self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
    score = 0
    if 'c2p' in self.pos_att_type:
        scale = tf.math.sqrt(tf.cast(shape_list(pos_key_layer)[-1] * scale_factor, tf.float32))
        c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 2, 1]))
        c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = take_along_axis(c2p_att, tf.broadcast_to(tf.squeeze(c2p_pos, 0), [shape_list(query_layer)[0], shape_list(query_layer)[1], shape_list(relative_pos)[-1]]))
        score += c2p_att / scale
    if 'p2c' in self.pos_att_type:
        scale = tf.math.sqrt(tf.cast(shape_list(pos_query_layer)[-1] * scale_factor, tf.float32))
        if shape_list(key_layer)[-2] != shape_list(query_layer)[-2]:
            r_pos = build_relative_position(shape_list(key_layer)[-2], shape_list(key_layer)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
            r_pos = tf.expand_dims(r_pos, 0)
        else:
            r_pos = relative_pos
        p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 2, 1]))
        p2c_att = tf.transpose(take_along_axis(p2c_att, tf.broadcast_to(tf.squeeze(p2c_pos, 0), [shape_list(query_layer)[0], shape_list(key_layer)[-2], shape_list(key_layer)[-2]])), [0, 2, 1])
        score += p2c_att / scale
    return score