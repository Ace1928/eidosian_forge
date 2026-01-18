import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref
import numpy as np
from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls
@property
def losses(self):
    """List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
      A list of tensors.
    """
    collected_losses = []
    for layer in self._flatten_layers():
        if layer._eager_losses:
            if layer._eager_losses[0] is not base_layer_utils.REVIVED_LOSS_PLACEHOLDER:
                collected_losses.extend(layer._eager_losses)
        else:
            collected_losses.extend(layer._losses)
        for regularizer in layer._callable_losses:
            loss_tensor = regularizer()
            if loss_tensor is not None:
                collected_losses.append(loss_tensor)
    return collected_losses