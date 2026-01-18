import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
def check_path_owner(path: str) -> bool:
    if sys.platform == 'win32' or not hasattr(os, 'geteuid'):
        return True
    assert os.path.isabs(path)
    previous = None
    while path != previous:
        if os.path.lexists(path):
            if os.geteuid() == 0:
                try:
                    path_uid = get_path_uid(path)
                except OSError:
                    return False
                return path_uid == 0
            else:
                return os.access(path, os.W_OK)
        else:
            previous, path = (path, os.path.dirname(path))
    return False