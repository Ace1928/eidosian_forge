from .util import (
import sys
from functools import reduce
def num_file_handles(self):
    """:return: amount of file handles in use. Each mapped region uses one file handle"""
    return self._handle_count