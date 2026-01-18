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
def is_tarfile(name):
    """Return True if name points to a tar archive that we
       are able to handle, else return False.

       'name' should be a string, file, or file-like object.
    """
    try:
        if hasattr(name, 'read'):
            pos = name.tell()
            t = open(fileobj=name)
            name.seek(pos)
        else:
            t = open(name)
        t.close()
        return True
    except TarError:
        return False