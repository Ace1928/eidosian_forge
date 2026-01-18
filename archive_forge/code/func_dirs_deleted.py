import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
@property
def dirs_deleted(self):
    """
        List of directories that were deleted.
        """
    return self._dirs_deleted