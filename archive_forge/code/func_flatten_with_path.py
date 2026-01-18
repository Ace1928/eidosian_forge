from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def flatten_with_path(structure):
    """Flattens a possibly nested structure into a list.

  This is a variant of :func:`~tree.flattens` which produces a list of
  pairs: ``(path, item)``.  A path is a tuple of indices and/or keys
  which uniquely identifies the position of the corresponding ``item``.

  >>> tree.flatten_with_path([{"foo": 42}])
  [((0, 'foo'), 42)]

  Args:
    structure: An arbitrarily nested structure.

  Returns:
    A list of ``(path, item)`` pairs corresponding to the flattened version
    of the input `structure`.

  Raises:
    TypeError:
      If ``structure`` is or contains a mapping with non-sortable keys.
  """
    return list(_yield_flat_up_to(structure, structure))