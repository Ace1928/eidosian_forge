import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_exception_on_communicate_error(self):
    mock = self.useFixture(fixtures.MockPatch('subprocess.Popen.communicate', side_effect=OSError(errno.EAGAIN, 'fake-test')))
    self.assertRaises(OSError, processutils.execute, '/usr/bin/env', 'false', check_exit_code=False)
    self.assertEqual(1, mock.mock.call_count)