from __future__ import unicode_literals
import errno
import sys
import os
import shutil
import os.path as op
from datetime import datetime
import stat
from send2trash.compat import text_type, environb
from send2trash.util import preprocess_paths
from send2trash.exceptions import TrashPermissionError
def info_for(src, topdir):
    if topdir is None or not is_parent(topdir, src):
        src = op.abspath(src)
    else:
        src = op.relpath(src, topdir)
    info = '[Trash Info]\n'
    info += 'Path=' + quote(src) + '\n'
    info += 'DeletionDate=' + format_date(datetime.now()) + '\n'
    return info