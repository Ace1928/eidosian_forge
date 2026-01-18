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
def convert_content_to_text(self):
    if len(self._value_tokens) == 1 and (not self._leading_whitespace_token) and (not self._trailing_whitespace_token) and isinstance(self._value_tokens[0], Deb822Token):
        return self._value_tokens[0].text
    return ''.join((t.text for t in self._iter_content_tokens()))