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
def test_cache_wrap(self):
    input = {'disk1': nt(100, 100, 100)}
    wrap_numbers(input, 'disk_io')
    input = {'disk1': nt(100, 100, 10)}
    wrap_numbers(input, 'disk_io')
    cache = wrap_numbers.cache_info()
    self.assertEqual(cache[0], {'disk_io': input})
    self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 100}})
    self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})

    def check_cache_info():
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 100}})
        self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})
    input = {'disk1': nt(100, 100, 10)}
    wrap_numbers(input, 'disk_io')
    cache = wrap_numbers.cache_info()
    self.assertEqual(cache[0], {'disk_io': input})
    check_cache_info()
    input = {'disk1': nt(100, 100, 90)}
    wrap_numbers(input, 'disk_io')
    cache = wrap_numbers.cache_info()
    self.assertEqual(cache[0], {'disk_io': input})
    check_cache_info()
    input = {'disk1': nt(100, 100, 20)}
    wrap_numbers(input, 'disk_io')
    cache = wrap_numbers.cache_info()
    self.assertEqual(cache[0], {'disk_io': input})
    self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 190}})
    self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})