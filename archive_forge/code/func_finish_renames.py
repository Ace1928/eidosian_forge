from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def finish_renames(self):
    for stamp, new_path in self._pending_renames:
        self._up_rename(stamp, new_path)
    self._pending_renames = []