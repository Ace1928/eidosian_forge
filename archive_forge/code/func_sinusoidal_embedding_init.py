from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def sinusoidal_embedding_init(shape, dtype=tf.float32) -> tf.Tensor:
    """Returns sinusoids for positional embedding"""
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(f'Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels.')
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = tf.exp(-log_timescale_increment * tf.range(channels // 2, dtype=tf.float32))
    scaled_time = tf.reshape(tf.range(length, dtype=tf.float32), (-1, 1)) * tf.reshape(inv_timescales, (1, -1))
    return tf.cast(tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1), dtype)