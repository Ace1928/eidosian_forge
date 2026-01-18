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
@traceback_utils.filter_traceback
def op_dispatch_handler(*args, **kwargs):
    """Call `dispatch_target`, peforming dispatch when appropriate."""
    if api_dispatcher is not None:
        if iterable_params is not None:
            args, kwargs = replace_iterable_params(args, kwargs, iterable_params)
        result = api_dispatcher.Dispatch(args, kwargs)
        if result is not NotImplemented:
            return result
    try:
        return dispatch_target(*args, **kwargs)
    except (TypeError, ValueError):
        result = dispatch(op_dispatch_handler, args, kwargs)
        if result is not OpDispatcher.NOT_SUPPORTED:
            return result
        else:
            raise