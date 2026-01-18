import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def save_to_filename(self, filename=None):
    if filename is None:
        filename = self._filename
    create_new = True
    read_only = False
    keep_cache_in_memory = False
    with self._fi as lib:
        multibitmap = lib.FreeImage_OpenMultiBitmap(self._ftype, efn(filename), create_new, read_only, keep_cache_in_memory, 0)
        multibitmap = ctypes.c_void_p(multibitmap)
        if not multibitmap:
            msg = 'Could not open file "%s" for writing multi-image: %s' % (self._filename, self._fi._get_error_message())
            raise ValueError(msg)
        self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))