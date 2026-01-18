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
def append_separator(self, space_after_separator=True):
    separator_token = self._default_separator_factory()
    if separator_token.is_whitespace:
        space_after_separator = False
    self._changed = True
    self._append_continuation_line_token_if_necessary()
    self._token_list.append(separator_token)
    if space_after_separator and (not separator_token.is_whitespace):
        self._token_list.append(Deb822WhitespaceToken(' '))