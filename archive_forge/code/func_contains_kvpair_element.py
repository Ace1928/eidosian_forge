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
def contains_kvpair_element(self, item):
    if not isinstance(item, (str, tuple, Deb822FieldNameToken)):
        return False
    item = cast('ParagraphKey', item)
    try:
        return self.get_kvpair_element(item, use_get=True) is not None
    except AmbiguousDeb822FieldKeyError:
        return True