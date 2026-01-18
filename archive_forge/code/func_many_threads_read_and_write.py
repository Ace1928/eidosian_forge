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
def many_threads_read_and_write():
    """
    >>> with open('f', 'w+b') as file:
    ...     _ = file.write(b'0\\n')
    >>> with open('f.lock', 'w+b') as file:
    ...     _ = file.write(b'0\\n')

    >>> n = 50
    >>> threads = [threading.Thread(target=inc) for i in range(n)]
    >>> _ = [thread.start() for thread in threads]
    >>> _ = [thread.join() for thread in threads]
    >>> with open('f', 'rb') as file:
    ...     saved = int(file.read().strip())
    >>> saved == n
    True

    >>> os.remove('f')

    We should only have one pid in the lock file:

    >>> f = open('f.lock')
    >>> len(f.read().strip().split())
    1
    >>> f.close()

    >>> os.remove('f.lock')

    """