import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
@run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
def encoder_train():
    loss = model(input_ids, labels=input_ids, training=True)[0]
    gradients = tf.gradients(loss, model.trainable_variables)
    return gradients