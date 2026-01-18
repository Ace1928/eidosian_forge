from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def storage_class(self):
    """
        Retrieves the storage class (if any) of the entity pointed at by the
        cursor.
        """
    if not hasattr(self, '_storage_class'):
        self._storage_class = conf.lib.clang_Cursor_getStorageClass(self)
    return StorageClass.from_id(self._storage_class)