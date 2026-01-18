from .util import (
import sys
from functools import reduce
def includes_ofs(self, ofs):
    """:return: True if the given absolute offset is contained in the cursors
            current region

        **Note:** cursor must be valid for this to work"""
    return self._region._b + self._ofs <= ofs < self._region._b + self._ofs + self._size