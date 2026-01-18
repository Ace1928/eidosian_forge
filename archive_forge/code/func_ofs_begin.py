from .util import (
import sys
from functools import reduce
def ofs_begin(self):
    """:return: offset to the first byte pointed to by our cursor

        **Note:** only if is_valid() is True"""
    return self._region._b + self._ofs