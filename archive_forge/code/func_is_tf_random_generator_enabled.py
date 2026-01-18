import collections
import itertools
import json
import os
import random
import sys
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.distribute import distribute_coordinator_utils as dc
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.engine import keras_tensor
from keras.src.utils import control_flow_util
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@keras_export('keras.backend.experimental.is_tf_random_generator_enabled', v1=[])
def is_tf_random_generator_enabled():
    """Check whether `tf.random.Generator` is used for RNG in Keras.

    Compared to existing TF stateful random ops, `tf.random.Generator` uses
    `tf.Variable` and stateless random ops to generate random numbers,
    which leads to better reproducibility in distributed training.
    Note enabling it might introduce some breakage to existing code,
    by producing differently-seeded random number sequences
    and breaking tests that rely on specific random numbers being generated.
    To disable the
    usage of `tf.random.Generator`, please use
    `tf.keras.backend.experimental.disable_random_generator`.

    We expect the `tf.random.Generator` code path to become the default, and
    will remove the legacy stateful random ops such as `tf.random.uniform` in
    the future (see the [TF RNG guide](
    https://www.tensorflow.org/guide/random_numbers)).

    This API will also be removed in a future release as well, together with
    `tf.keras.backend.experimental.enable_tf_random_generator()` and
    `tf.keras.backend.experimental.disable_tf_random_generator()`

    Returns:
      boolean: whether `tf.random.Generator` is used for random number
        generation in Keras.
    """
    return _USE_GENERATOR_FOR_RNG