import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def flatten_with_joined_string_paths(structure, separator='/', expand_composites=False):
    """Returns a list of (string path, atom) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each atom was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of (string, atom) tuples.
  """
    flat_paths = yield_flat_paths(structure, expand_composites=expand_composites)

    def stringify_and_join(path_elements):
        return separator.join((str(path_element) for path_element in path_elements))
    flat_string_paths = (stringify_and_join(path) for path in flat_paths)
    return list(zip(flat_string_paths, flatten(structure, expand_composites=expand_composites)))