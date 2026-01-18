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
def test_interthread_external_lock(self):
    call_list = []

    @lockutils.synchronized('foo', external=True, lock_path=self.lock_dir)
    def foo(param):
        """Simulate a long-running threaded operation."""
        call_list.append(param)
        time.sleep(0.5)
        call_list.append(param)

    def other(param):
        foo(param)
    thread = threading.Thread(target=other, args=('other',))
    thread.start()
    start = time.time()
    while not os.path.exists(os.path.join(self.lock_dir, 'foo')):
        if time.time() - start > 5:
            self.fail('Timed out waiting for thread to grab lock')
        time.sleep(0)
    thread1 = threading.Thread(target=other, args=('main',))
    thread1.start()
    thread1.join()
    thread.join()
    self.assertEqual(['other', 'other', 'main', 'main'], call_list)