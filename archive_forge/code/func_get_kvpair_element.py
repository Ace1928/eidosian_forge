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
def get_kvpair_element(self, item, use_get=False):
    key, index, name_token = _unpack_key(item)
    if use_get:
        nodes = self._kvpair_elements.get(key)
        if nodes is None:
            return None
    else:
        nodes = self._kvpair_elements[key]
    node = self._resolve_to_single_node(nodes, key, index, name_token, use_get=use_get)
    if node is not None:
        return node.value
    return None