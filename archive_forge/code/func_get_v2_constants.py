from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_v2_constants(module: Any) -> Sequence[str]:
    """Get a list of TF 2.0 constants in this module.

  Args:
    module: TensorFlow module.

  Returns:
    List of all API constants under the given module including TensorFlow and
    Estimator constants.
  """
    constants_v2 = []
    tensorflow_constants_attr = API_ATTRS[TENSORFLOW_API_NAME].constants
    estimator_constants_attr = API_ATTRS[ESTIMATOR_API_NAME].constants
    if hasattr(module, tensorflow_constants_attr):
        constants_v2.extend(getattr(module, tensorflow_constants_attr))
    if hasattr(module, estimator_constants_attr):
        constants_v2.extend(getattr(module, estimator_constants_attr))
    return constants_v2