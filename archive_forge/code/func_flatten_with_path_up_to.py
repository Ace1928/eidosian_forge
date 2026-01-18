from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def flatten_with_path_up_to(shallow_structure, input_structure, check_types=True):
    """Flattens `input_structure` up to `shallow_structure`.

  This is a combination of :func:`~tree.flatten_up_to` and
  :func:`~tree.flatten_with_path`

  Args:
    shallow_structure: A structure with the same (but possibly more shallow)
      layout as `input_structure`.
    input_structure: An arbitrarily nested structure.
    check_types: If `True`, check that each node in shallow_tree has the
      same type as the corresponding node in `input_structure`.

  Returns:
    A list of ``(path, item)`` pairs corresponding to the partially flattened
    version of `input_structure` wrt `shallow_structure`.

  Raises:
    TypeError: If the layout of `shallow_structure` does not match that of
      `input_structure`.
    TypeError: If `input_structure` is or contains a mapping with non-sortable
      keys.
    TypeError: If `check_types` is `True` and `shallow_structure` and
      `input_structure` differ in the types of their components.
  """
    _assert_shallow_structure(shallow_structure, input_structure, path=(), check_types=check_types)
    return list(_yield_flat_up_to(shallow_structure, input_structure))