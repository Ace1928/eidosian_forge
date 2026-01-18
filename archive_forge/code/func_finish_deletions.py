from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def finish_deletions(self):
    if self._pending_deletions:
        for relpath in reversed(self._pending_deletions):
            self._up_rmdir(relpath)
        self._pending_deletions = []