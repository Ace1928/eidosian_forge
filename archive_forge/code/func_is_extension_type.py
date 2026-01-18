import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
def is_extension_type(tensor):
    """Returns whether a tensor is of an ExtensionType.

  github.com/tensorflow/community/pull/269
  Currently it works by checking if `tensor` is a `CompositeTensor` instance,
  but this will be changed to use an appropriate extensiontype protocol
  check once ExtensionType is made public.

  Args:
    tensor: An object to test

  Returns:
    True if the tensor is an extension type object, false if not.
  """
    return isinstance(tensor, composite_tensor.CompositeTensor)