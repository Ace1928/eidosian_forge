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
def check_undecodable_bytes_error(self, binary):
    out_bytes = b'out: password="secret1" ' + UNDECODABLE_BYTES
    err_bytes = b'err: password="secret2" ' + UNDECODABLE_BYTES
    conn = FakeSshConnection(1, out=out_bytes, err=err_bytes)
    out_bytes = b'out: password="***" ' + UNDECODABLE_BYTES
    err_bytes = b'err: password="***" ' + UNDECODABLE_BYTES
    exc = self.assertRaises(processutils.ProcessExecutionError, processutils.ssh_execute, conn, 'ls', binary=binary, check_exit_code=True)
    out = exc.stdout
    err = exc.stderr
    self.assertEqual(os.fsdecode(out_bytes), out)
    self.assertEqual(os.fsdecode(err_bytes), err)