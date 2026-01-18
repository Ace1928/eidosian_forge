from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
def hf_compute_loss(self, logits, labels):
    label_interp_shape = shape_list(labels)[1:]
    upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method='bilinear')
    loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def masked_loss(real, pred):
        unmasked_loss = loss_fct(real, pred)
        mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        return tf.reshape(reduced_masked_loss, (1,))
    return masked_loss(labels, upsampled_logits)