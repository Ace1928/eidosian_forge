from __future__ import absolute_import, print_function, unicode_literals
import typing
import collections
import contextlib
import pkg_resources
from ..errors import ResourceReadOnly
from .base import Opener
from .errors import EntryPointError, UnsupportedProtocol
from .parse import parse_fs_url
@property
def protocols(self):
    """`list`: the list of supported protocols."""
    _protocols = list(self._protocols)
    if self.load_extern:
        _protocols.extend((entry_point.name for entry_point in pkg_resources.iter_entry_points('fs.opener')))
        _protocols = list(collections.OrderedDict.fromkeys(_protocols))
    return _protocols