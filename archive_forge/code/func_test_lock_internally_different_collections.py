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
def test_lock_internally_different_collections(self):
    s1 = lockutils.Semaphores()
    s2 = lockutils.Semaphores()
    trigger = threading.Event()
    who_ran = collections.deque()

    def f(name, semaphores, pull_trigger):
        with lockutils.internal_lock('testing', semaphores=semaphores):
            if pull_trigger:
                trigger.set()
            else:
                trigger.wait()
            who_ran.append(name)
    threads = [threading.Thread(target=f, args=(1, s1, True)), threading.Thread(target=f, args=(2, s2, False))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    self.assertEqual([1, 2], sorted(who_ran))