import abc
import types
import warnings
import numpy as np
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Computes how often integer targets are in the top `K` predictions.

  Standalone usage:
  >>> y_true = [2, 1]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
  ...     y_true, y_pred, k=3)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([1., 1.], dtype=float32)

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.

  Returns:
    Sparse top K categorical accuracy value.
  """
    y_pred_rank = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred).shape.ndims
    y_true_rank = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_true).shape.ndims
    if y_true_rank is not None and y_pred_rank is not None:
        if y_pred_rank > 2:
            y_pred = array_ops.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            y_true = array_ops.reshape(y_true, [-1])
    return math_ops.cast(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), backend.floatx())