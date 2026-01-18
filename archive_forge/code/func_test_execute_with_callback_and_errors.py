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
@mock.patch.object(subprocess.Popen, 'communicate')
def test_execute_with_callback_and_errors(self, mock_comm):
    on_execute_callback = mock.Mock()
    on_completion_callback = mock.Mock()

    def fake_communicate(*args, timeout=None):
        raise IOError('Broken pipe')
    mock_comm.side_effect = fake_communicate
    self.assertRaises(IOError, processutils.execute, TRUE_UTILITY, on_execute=on_execute_callback, on_completion=on_completion_callback)
    self.assertEqual(1, on_execute_callback.call_count)
    self.assertEqual(1, on_completion_callback.call_count)