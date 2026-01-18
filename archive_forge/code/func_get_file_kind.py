from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def get_file_kind(self, file_id):
    return self._byid[file_id].kind