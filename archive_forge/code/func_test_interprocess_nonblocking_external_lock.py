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
def test_interprocess_nonblocking_external_lock(self):
    """Check that we're not actually blocking between processes."""
    nb_calls = multiprocessing.Value('i', 0)

    @lockutils.synchronized('foo', blocking=False, external=True, lock_path=self.lock_dir)
    def foo(param):
        """Simulate a long-running operation in a process."""
        param.value += 1
        time.sleep(0.5)

    def other(param):
        foo(param)
    process = multiprocessing.Process(target=other, args=(nb_calls,))
    process.start()
    start = time.time()
    while not os.path.exists(os.path.join(self.lock_dir, 'foo')):
        if time.time() - start > 5:
            self.fail('Timed out waiting for process to grab lock')
        time.sleep(0)
    process1 = multiprocessing.Process(target=other, args=(nb_calls,))
    process1.start()
    process1.join()
    process.join()
    self.assertEqual(1, nb_calls.value)