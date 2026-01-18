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
def test_check_cwd(self):
    tmpdir = tempfile.mkdtemp()
    out, err = processutils.execute('/usr/bin/env', 'sh', '-c', 'pwd', cwd=tmpdir)
    self.assertIn(tmpdir, out)