import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
def stat_info(self, path):
    """
        Returns a stat information object for the specified path from
        the snapshot.

        Attached information is subject to change. Do not use unless
        you specify `stat` in constructor. Use :func:`inode`, :func:`mtime`,
        :func:`isdir` instead.

        :param path:
            The path for which stat information should be obtained
            from a snapshot.
        """
    return self._stat_info[path]