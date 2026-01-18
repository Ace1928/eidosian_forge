from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def map_structure_with_path_up_to(shallow_structure, func, *structures, **kwargs):
    """Maps `func` through given structures up to `shallow_structure`.

  This is a combination of :func:`~tree.map_structure_up_to` and
  :func:`~tree.map_structure_with_path`

  Args:
    shallow_structure: A structure with layout common to all `structures`.
    func: A callable that accepts a path and as many arguments as there are
      structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: No valid keyword arguments.

  Raises:
    ValueError: If `func` is not callable or if `structures` have different
      layout or if the layout of `shallow_structure` does not match that of
      `structures` or if no structures were given.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout
    as `shallow_tree`.
  """
    if 'check_types' in kwargs:
        logging.warning('The use of `check_types` is deprecated and does not have any effect.')
    del kwargs
    results = []
    for path_and_values in _multiyield_flat_up_to(shallow_structure, *structures):
        results.append(func(*path_and_values))
    return unflatten_as(shallow_structure, results)