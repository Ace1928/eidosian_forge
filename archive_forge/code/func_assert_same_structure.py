from tensorflow.python.util import nest_util
def assert_same_structure(nest1, nest2, check_types=True):
    """Asserts that two structures are nested in the same way.

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences should be same as
      well. For dictionary, "type" of dictionary is considered to include its
      keys. In other words, two dictionaries with different keys are considered
      to have a different "type". If set to `False`, two iterables are
      considered same as long as they yield the elements that have same
      structures.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
    nest_util.assert_same_structure(nest_util.Modality.DATA, nest1, nest2, check_types)