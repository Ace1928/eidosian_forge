import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def flatten_with_tuple_paths(structure, expand_composites=False):
    """Returns a list of `(tuple_path, atom)` tuples.

  The order of pairs produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each atom was located. See `nest.yield_flat_paths`
  for more information about tuple paths.

  Args:
    structure: the nested structure to flatten.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of `(tuple_path, atom)` tuples. Each `tuple_path` is a tuple
    of indices and/or dictionary keys that uniquely specify the path to
    `atom` within `structure`.
  """
    return list(zip(yield_flat_paths(structure, expand_composites=expand_composites), flatten(structure, expand_composites=expand_composites)))