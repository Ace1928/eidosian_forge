import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
def test_lock_internal_fair(self):
    """Check that we're actually fair."""

    def f(_id):
        with lockutils.lock('testlock', 'test-', external=False, fair=True):
            lock_holder.append(_id)
    lock_holder = []
    threads = []
    with lockutils.lock('testlock', 'test-', external=False, fair=True):
        for i in range(10):
            thread = threading.Thread(target=f, args=(i,))
            threads.append(thread)
            thread.start()
            time.sleep(0.5)
    for thread in threads:
        thread.join()
    self.assertEqual(10, len(lock_holder))
    for i in range(10):
        self.assertEqual(i, lock_holder[i])