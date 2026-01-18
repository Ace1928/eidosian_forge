import doctest
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from zope.testing import setupstack
import zc.lockfile
def pid_in_lockfile():
    """
    >>> import os, zc.lockfile
    >>> pid = os.getpid()
    >>> lock = zc.lockfile.LockFile("f.lock")
    >>> f = open("f.lock")
    >>> _ = f.seek(1)
    >>> f.read().strip() == str(pid)
    True
    >>> f.close()

    Make sure that locking twice does not overwrite the old pid:

    >>> lock = zc.lockfile.LockFile("f.lock")
    Traceback (most recent call last):
      ...
    zc.lockfile.LockError: Couldn't lock 'f.lock'

    >>> f = open("f.lock")
    >>> _ = f.seek(1)
    >>> f.read().strip() == str(pid)
    True
    >>> f.close()

    >>> lock.close()
    """