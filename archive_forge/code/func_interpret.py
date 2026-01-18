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
def interpret(self, kvpair_element, discard_comments_on_read=True):
    token_list = []
    token_list.extend(self._parse_kvpair(kvpair_element))
    return self._high_level_interpretation(kvpair_element, token_list, discard_comments_on_read=discard_comments_on_read)