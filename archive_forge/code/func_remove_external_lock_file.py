import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def remove_external_lock_file(name, lock_file_prefix=None, lock_path=None, semaphores=None):
    """Remove an external lock file when it's not used anymore
    This will be helpful when we have a lot of lock files
    """
    with internal_lock(name, semaphores=semaphores):
        lock_file_path = _get_lock_path(name, lock_file_prefix, lock_path)
        try:
            os.remove(lock_file_path)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                LOG.warning('Failed to remove file %(file)s', {'file': lock_file_path})