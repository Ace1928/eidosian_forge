import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.nest.get_traverse_shallow_structure', v1=[])
def get_traverse_shallow_structure(traverse_fn, structure, expand_composites=False):
    """Generates a shallow structure from a `traverse_fn` and `structure`.

  `traverse_fn` must accept any possible subtree of `structure` and return
  a depth=1 structure containing `True` or `False` values, describing which
  of the top-level subtrees may be traversed.  It may also
  return scalar `True` or `False` "traversal is OK / not OK for all subtrees."

  Examples are available in the unit tests (nest_test.py).

  Args:
    traverse_fn: Function taking a substructure and returning either a scalar
      `bool` (whether to traverse that substructure or not) or a depth=1
      shallow structure of the same type, describing which parts of the
      substructure to traverse.
    structure: The structure to traverse.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A shallow structure containing python bools, which can be passed to
    `map_structure_up_to` and `flatten_up_to`.

  Raises:
    TypeError: if `traverse_fn` returns a nested structure for an atom input.
      or a structure with depth higher than 1 for a nested structure input,
      or if any leaf values in the returned structure or scalar are not type
      `bool`.
  """
    is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
    to_traverse = traverse_fn(structure)
    if not is_nested_fn(structure):
        if not isinstance(to_traverse, bool):
            raise TypeError('traverse_fn returned structure: %s for non-structure: %s' % (to_traverse, structure))
        return to_traverse
    level_traverse = []
    if isinstance(to_traverse, bool):
        if not to_traverse:
            return False
        else:
            for branch in nest_util.yield_value(nest_util.Modality.CORE, structure):
                level_traverse.append(get_traverse_shallow_structure(traverse_fn, branch, expand_composites=expand_composites))
    elif not is_nested_fn(to_traverse):
        raise TypeError('traverse_fn returned a non-bool scalar: %s for input: %s' % (to_traverse, structure))
    else:
        assert_shallow_structure(to_traverse, structure, expand_composites=expand_composites)
        for t, branch in zip(nest_util.yield_value(nest_util.Modality.CORE, to_traverse), nest_util.yield_value(nest_util.Modality.CORE, structure)):
            if not isinstance(t, bool):
                raise TypeError("traverse_fn didn't return a depth=1 structure of bools.  saw: %s  for structure: %s" % (to_traverse, structure))
            if t:
                level_traverse.append(get_traverse_shallow_structure(traverse_fn, branch))
            else:
                level_traverse.append(False)
    return nest_util.sequence_like(structure, level_traverse)