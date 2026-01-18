import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def flatten_with_tuple_paths_up_to(shallow_tree, input_tree, check_types=True, expand_composites=False):
    """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as structures in
  the partially flattened output.

  Returns a list of (path, value) pairs, where value a leaf node in the
  flattened tree, and path is the tuple path of that leaf in input_tree.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-item list: `[((), input_tree)]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure layout
  as `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                        input_tree)
  flattened_shallow_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                          shallow_tree)

  # Output is:
  # [((0, 0), [2, 2]),
  #  ((0, 1), [3, 3]),
  #  ((1, 0), [4, 9]),
  #  ((1, 1), [5, 5])]
  #
  # [((0, 0), True),
  #  ((0, 1), True),
  #  ((1, 0), False),
  #  ((1, 1), True)]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [((0, 0), ('a', 1)),
  #  ((0, 1, 0), ('b', 2)),
  #  ((0, 1, 1, 0), ('c', 3)),
  #  ((0, 1, 1, 1), ('d', 4))]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_with_tuple_paths_up_to(0, 0)  # Output: [(), 0]

  flatten_with_tuple_paths_up_to(0, [0, 1, 2])  # Output: [(), [0, 1, 2]]

  flatten_with_tuple_paths_up_to([0, 1, 2], 0)  # Output: TypeError

  flatten_with_tuple_paths_up_to([0, 1, 2], [0, 1, 2])
  # Output: [((0,) 0), ((1,), 1), ((2,), 2)]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an atom or a nested structure.
      Note, numpy arrays are considered atoms.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a nested structure but `input_tree` is not.
    TypeError: If the structure types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the structure lengths of `shallow_tree` are different from
      `input_tree`.
  """
    is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
    assert_shallow_structure(shallow_tree, input_tree, check_types=check_types, expand_composites=expand_composites)
    return list(nest_util.yield_flat_up_to(nest_util.Modality.CORE, shallow_tree, input_tree, is_nested_fn))