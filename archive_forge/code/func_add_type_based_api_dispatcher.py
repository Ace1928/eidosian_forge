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
def add_type_based_api_dispatcher(target):
    """Adds a PythonAPIDispatcher to the given TensorFlow API function."""
    if hasattr(target, TYPE_BASED_DISPATCH_ATTR):
        raise ValueError(f'{target} already has a type-based API dispatcher.')
    _, unwrapped = tf_decorator.unwrap(target)
    target_argspec = tf_inspect.getargspec(unwrapped)
    if target_argspec.varargs or target_argspec.keywords:
        return target
    setattr(target, TYPE_BASED_DISPATCH_ATTR, _api_dispatcher.PythonAPIDispatcher(unwrapped.__name__, target_argspec.args, target_argspec.defaults))
    _TYPE_BASED_DISPATCH_SIGNATURES[target] = collections.defaultdict(list)
    return target