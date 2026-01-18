from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
@staticmethod
def get_buf_from_file(f):
    if hasattr(f, 'getbuffer'):
        return f.getbuffer()
    elif hasattr(f, 'getvalue'):
        return f.getvalue()
    else:
        try:
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            return b''