import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
def can_symlink():
    """Return whether running process can create symlinks."""
    if sys.platform != 'win32':
        return True
    test_source = tempfile.mkdtemp()
    test_target = test_source + 'can_symlink'
    try:
        os.symlink(test_source, test_target)
    except (NotImplementedError, OSError):
        return False
    return True