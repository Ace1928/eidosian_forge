import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def append_newline(self):
    if self._previous_is_newline():
        raise ValueError('Cannot add a newline after a token that ends on a newline')
    self._token_list.append(Deb822NewlineAfterValueToken())