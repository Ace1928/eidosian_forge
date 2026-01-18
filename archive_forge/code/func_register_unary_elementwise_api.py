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
def register_unary_elementwise_api(func):
    """Decorator that registers a TensorFlow op as a unary elementwise API."""
    _UNARY_ELEMENTWISE_APIS.append(func)
    for args, handler in _ELEMENTWISE_API_HANDLERS.items():
        if len(args) == 1:
            _add_dispatch_for_unary_elementwise_api(func, args[0], handler)
    return func