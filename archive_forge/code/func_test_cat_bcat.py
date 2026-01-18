import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
def test_cat_bcat(self):
    testfn = self.get_testfn()
    with open(testfn, 'w') as f:
        f.write('foo')
    self.assertEqual(cat(testfn), 'foo')
    self.assertEqual(bcat(testfn), b'foo')
    self.assertRaises(FileNotFoundError, cat, testfn + '-invalid')
    self.assertRaises(FileNotFoundError, bcat, testfn + '-invalid')
    self.assertEqual(cat(testfn + '-invalid', fallback='bar'), 'bar')
    self.assertEqual(bcat(testfn + '-invalid', fallback='bar'), 'bar')