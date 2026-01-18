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
def convert_to_text_without_comments(self):
    if self._text_no_comments_cached is None:
        self._text_no_comments_cached = ''.join((t.text for t in self.iter_tokens() if not t.is_comment))
    return self._text_no_comments_cached