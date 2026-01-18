from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def get_matching_input_branch(shallow_key):
    for input_key, input_branch in input_iter:
        if input_key == shallow_key:
            return input_branch
    raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))