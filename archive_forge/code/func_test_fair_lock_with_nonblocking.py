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
def test_fair_lock_with_nonblocking(self):

    def do_test():
        with lockutils.lock('testlock', 'test-', fair=True, blocking=False):
            pass
    self.assertRaises(NotImplementedError, do_test)