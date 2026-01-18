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
def find_ext_volume_fallback_trash(volume_root):
    trash_dir = op.join(volume_root, TOPDIR_FALLBACK)
    try:
        check_create(trash_dir)
    except OSError as e:
        if e.errno == errno.EACCES:
            raise TrashPermissionError(e.filename)
        raise
    return trash_dir