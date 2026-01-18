from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
@property
def is_removed(self):
    return self.action == FILE_ACTION_REMOVED