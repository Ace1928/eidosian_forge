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
def test_exception_and_masking(self):
    tmpfilename = self.create_tempfiles([['test_exceptions_and_masking', TEST_EXCEPTION_AND_MASKING_SCRIPT]], ext='bash')[0]
    os.chmod(tmpfilename, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    err = self.assertRaises(processutils.ProcessExecutionError, processutils.execute, tmpfilename, 'password="secret"', 'something')
    self.assertEqual(38, err.exit_code)
    self.assertIsInstance(err.stdout, str)
    self.assertIsInstance(err.stderr, str)
    self.assertIn('onstdout --password="***"', err.stdout)
    self.assertIn('onstderr --password="***"', err.stderr)
    self.assertEqual(' '.join([tmpfilename, 'password="***"', 'something']), err.cmd)
    self.assertNotIn('secret', str(err))