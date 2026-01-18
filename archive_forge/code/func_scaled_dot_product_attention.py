from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(shape_list(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += tf.cast(mask * -10000.0, dtype=scaled_attention_logits.dtype)
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, dtype=scaled_attention_logits.dtype)
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = stable_softmax(scaled_attention_logits, axis=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = tf.matmul(attention_weights, v)
    return (output, attention_weights)