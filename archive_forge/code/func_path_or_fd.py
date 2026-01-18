from .util import (
import sys
from functools import reduce
def path_or_fd(self):
    """:return: path or file descriptor of the underlying mapped file"""
    return self._rlist.path_or_fd()