import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def is_nested_or_composite(seq):
    """Returns true if its input is a nested structure or a composite.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a nested structure.

  Args:
    seq: the value to test.

  Returns:
    True if the input is a nested structure or a composite.
  """
    return _is_nested_or_composite(seq)