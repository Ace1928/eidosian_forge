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
def test_get_override(self):
    lockutils._register_opts(self.conf)
    self.conf.set_override('lock_path', '/alternate/path', group='oslo_concurrency')
    self.assertEqual('/alternate/path', lockutils.get_lock_path(self.conf))