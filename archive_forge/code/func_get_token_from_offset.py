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
def get_token_from_offset(self, offset):
    """
    Returns the token containing the given character offset (0-based position in source text),
    or the preceeding token if the position is between tokens.
    """
    return self._tokens[bisect.bisect(self._token_offsets, offset) - 1]