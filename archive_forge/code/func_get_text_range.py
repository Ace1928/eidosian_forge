import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
def get_text_range(self, node, padded=True):
    """
    Returns the (startpos, endpos) positions in source text corresponding to the given node.
    Returns (0, 0) for nodes (like `Load`) that don't correspond to any particular text.

    See ``get_text_positions()`` for details on the ``padded`` argument.
    """
    start, end = self.get_text_positions(node, padded)
    return (self._line_numbers.line_to_offset(*start), self._line_numbers.line_to_offset(*end))