import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
@property
def files_deleted(self):
    """List of files that were deleted."""
    return self._files_deleted