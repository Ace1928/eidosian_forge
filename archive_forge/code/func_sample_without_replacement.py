import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
def sample_without_replacement(logits, num_samples):
    """
    categorical sampling without replacement is currently not implemented the gumbel-max trick will do for now see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(shape_list(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices