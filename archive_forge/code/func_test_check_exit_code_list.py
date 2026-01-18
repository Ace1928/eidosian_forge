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
def test_check_exit_code_list(self):
    processutils.execute('/usr/bin/env', 'sh', '-c', 'exit 101', check_exit_code=(101, 102))
    processutils.execute('/usr/bin/env', 'sh', '-c', 'exit 102', check_exit_code=(101, 102))
    self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'sh', '-c', 'exit 103', check_exit_code=(101, 102))
    self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'sh', '-c', 'exit 0', check_exit_code=(101, 102))