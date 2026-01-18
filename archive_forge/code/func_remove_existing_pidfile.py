from __future__ import absolute_import
import errno
import os
import time
from . import (LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock,
def remove_existing_pidfile(pidfile_path):
    """ Remove the named PID file if it exists.

        Removing a PID file that doesn't already exist puts us in the
        desired state, so we ignore the condition if the file does not
        exist.

        """
    try:
        os.remove(pidfile_path)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            pass
        else:
            raise