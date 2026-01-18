import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.nest.is_attrs', v1=[])
def is_attrs(obj):
    """Returns a true if its input is an instance of an attr.s decorated class."""
    return _is_attrs(obj)