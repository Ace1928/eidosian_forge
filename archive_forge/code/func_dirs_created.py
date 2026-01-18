import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
@property
def dirs_created(self):
    """
        List of directories that were created.
        """
    return self._dirs_created