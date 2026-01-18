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
@property
def asttokens(self):
    if self._asttokens is None:
        self._asttokens = ASTTokens(self._text, tree=self.tree, filename=self._filename)
    return self._asttokens