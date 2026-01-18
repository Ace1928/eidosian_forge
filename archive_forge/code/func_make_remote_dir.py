from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def make_remote_dir(self, relpath, mode=None):
    if mode is None:
        mode = 509
    self._up_mkdir(relpath, mode)