from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def create_ustar_header(self, info, encoding, errors):
    """Return the object as a ustar header block.
        """
    info['magic'] = POSIX_MAGIC
    if len(info['linkname'].encode(encoding, errors)) > LENGTH_LINK:
        raise ValueError('linkname is too long')
    if len(info['name'].encode(encoding, errors)) > LENGTH_NAME:
        info['prefix'], info['name'] = self._posix_split_name(info['name'], encoding, errors)
    return self._create_header(info, USTAR_FORMAT, encoding, errors)