from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def delete_remote_dir_maybe(self, relpath):
    """Try to delete relpath, keeping failures to retry later."""
    try:
        self._up_rmdir(relpath)
    except errors.PathError:
        self._pending_deletions.append(relpath)