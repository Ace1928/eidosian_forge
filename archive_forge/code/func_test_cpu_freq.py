import contextlib
import datetime
import errno
import os
import platform
import pprint
import shutil
import signal
import socket
import sys
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import DEVNULL
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import IS_64BIT
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import enum
from psutil.tests import mock
from psutil.tests import retry_on_failure
@unittest.skipIf(MACOS and platform.machine() == 'arm64', 'skipped due to #1892')
@unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
def test_cpu_freq(self):

    def check_ls(ls):
        for nt in ls:
            self.assertEqual(nt._fields, ('current', 'min', 'max'))
            if nt.max != 0.0:
                self.assertLessEqual(nt.current, nt.max)
            for name in nt._fields:
                value = getattr(nt, name)
                self.assertIsInstance(value, (int, long, float))
                self.assertGreaterEqual(value, 0)
    ls = psutil.cpu_freq(percpu=True)
    if FREEBSD and (not ls):
        raise self.skipTest('returns empty list on FreeBSD')
    assert ls, ls
    check_ls([psutil.cpu_freq(percpu=False)])
    if LINUX:
        self.assertEqual(len(ls), psutil.cpu_count())