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
def test_lock_with_prefix(self):
    self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
    foo = lockutils.lock_with_prefix('mypfix-')
    with foo('mylock', external=True):
        pass