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
def remove_kvpair_element(self, key):
    key, idx, name_token = _unpack_key(key)
    field_list = self._kvpair_elements[key]
    if name_token is None and idx is None:
        for node in field_list:
            node.value.parent_element = None
            self._kvpair_order.remove_node(node)
        del self._kvpair_elements[key]
        return
    if name_token is not None:
        original_node = self._find_node_via_name_token(name_token, field_list)
        if original_node is None:
            msg = 'The field "{key}" is present but key used to access it is not.'
            raise KeyError(msg.format(key=key))
        node = original_node
    else:
        assert idx is not None
        try:
            node = field_list[idx]
        except KeyError:
            msg = 'The field "{key}" is present, but the index "{idx}" was invalid.'
            raise KeyError(msg.format(key=key, idx=idx))
    if len(field_list) == 1:
        del self._kvpair_elements[key]
    else:
        field_list.remove(node)
    node.value.parent_element = None
    self._kvpair_order.remove_node(node)