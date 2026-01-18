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
@classmethod
def from_kvpairs(cls, kvpair_elements):
    if not kvpair_elements:
        raise ValueError('A paragraph must consist of at least one field/value pair')
    kvpair_order = OrderedSet((kv.field_name for kv in kvpair_elements))
    if len(kvpair_order) == len(kvpair_elements):
        return Deb822NoDuplicateFieldsParagraphElement(kvpair_elements, kvpair_order)
    return Deb822DuplicateFieldsParagraphElement(kvpair_elements)