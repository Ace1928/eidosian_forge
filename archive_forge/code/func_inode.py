import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
def inode(self, path):
    """ Returns an id for path. """
    st = self._stat_info[path]
    return (st.st_ino, st.st_dev)