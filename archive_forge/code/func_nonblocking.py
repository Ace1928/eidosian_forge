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
@contextlib.contextmanager
def nonblocking(lock):
    """Try to acquire the internal lock without blocking."""
    if not lock.acquire(blocking=False):
        raise AcquireLockFailedException(name)
    try:
        yield lock
    finally:
        lock.release()