import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def register_binary_elementwise_assert_api(func):
    """Decorator that registers a TensorFlow op as a binary elementwise assert API.

  Different from `dispatch_for_binary_elementwise_apis`, this decorator is used
  for assert apis, such as assert_equal, assert_none_equal, etc, which return
  None in eager mode and an op in graph mode.

  Args:
    func: The function that implements the binary elementwise assert API.

  Returns:
    `func`
  """
    _BINARY_ELEMENTWISE_ASSERT_APIS.append(func)
    for args, handler in _ELEMENTWISE_API_HANDLERS.items():
        if len(args) == 3 and args[2] is _ASSERT_API_TAG:
            _add_dispatch_for_binary_elementwise_api(func, args[0], args[1], handler)
    return func