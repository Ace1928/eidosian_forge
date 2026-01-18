from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def get_recurse(self, root: MutableMapping[int, str | MutableMapping[int, str | MutableMapping[int, str]]] | typing.Literal['mouse', 'sgrmouse'], keys: Collection[int], more_available: bool):
    if not isinstance(root, MutableMapping):
        if root == 'mouse':
            return self.read_mouse_info(keys, more_available)
        if root == 'sgrmouse':
            return self.read_sgrmouse_info(keys, more_available)
        return (root, keys)
    if not keys:
        if more_available:
            raise MoreInputRequired()
        return None
    if keys[0] not in root:
        return None
    return self.get_recurse(root[keys[0]], keys[1:], more_available)