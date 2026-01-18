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
def test_lock_internally(self):
    """We can lock across multiple threads."""
    saved_sem_num = len(lockutils._semaphores)
    seen_threads = list()

    def f(_id):
        with lockutils.lock('testlock2', 'test-', external=False):
            for x in range(10):
                seen_threads.append(_id)
    threads = []
    for i in range(10):
        thread = threading.Thread(target=f, args=(i,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    self.assertEqual(100, len(seen_threads))
    for i in range(10):
        for j in range(9):
            self.assertEqual(seen_threads[i * 10], seen_threads[i * 10 + 1 + j])
    self.assertEqual(saved_sem_num, len(lockutils._semaphores), 'Semaphore leak detected')