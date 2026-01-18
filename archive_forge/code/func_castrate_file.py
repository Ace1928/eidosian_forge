from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def castrate_file(path, st):
    if not is_cython_generated_file(path, allow_failed=True, if_not_found=False):
        return
    try:
        f = open_new_file(path)
    except EnvironmentError:
        pass
    else:
        f.write('#error Do not use this file, it is the result of a failed Cython compilation.\n')
        f.close()
        if st:
            os.utime(path, (st.st_atime, st.st_mtime - 1))