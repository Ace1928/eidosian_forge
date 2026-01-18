import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def ensure_config_dir_exists(path=None):
    """Make sure a configuration directory exists.

    This makes sure that the directory exists.
    On windows, since configuration directories are 2 levels deep,
    it makes sure both the directory and the parent directory exists.
    """
    if path is None:
        path = config_dir()
    if not os.path.isdir(path):
        parent_dir = os.path.dirname(path)
        if not os.path.isdir(parent_dir):
            trace.mutter('creating config parent directory: %r', parent_dir)
            os.mkdir(parent_dir)
            osutils.copy_ownership_from_path(parent_dir)
        trace.mutter('creating config directory: %r', path)
        os.mkdir(path)
        osutils.copy_ownership_from_path(path)