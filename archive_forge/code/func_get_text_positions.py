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
def get_text_positions(self, node, padded):
    """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
    if getattr(node, '_broken_positions', None):
        return ((1, 0), (1, 0))
    if supports_tokenless(node):
        return self._get_text_positions_tokenless(node, padded)
    return self.asttokens.get_text_positions(node, padded)