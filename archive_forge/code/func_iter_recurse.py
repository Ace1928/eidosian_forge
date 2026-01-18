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
def iter_recurse(self, *, only_element_or_token_type=None):
    for part in self.iter_parts():
        if only_element_or_token_type is None or isinstance(part, only_element_or_token_type):
            yield cast('TE', part)
        if isinstance(part, Deb822Element):
            yield from part.iter_recurse(only_element_or_token_type=only_element_or_token_type)