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
@unittest.skipIf(LINUX and (not os.path.exists('/proc/diskstats')), '/proc/diskstats not available on this linux version')
@unittest.skipIf(CI_TESTING and (not psutil.disk_io_counters()), 'unreliable on CI')
def test_disk_io_counters(self):

    def check_ntuple(nt):
        self.assertEqual(nt[0], nt.read_count)
        self.assertEqual(nt[1], nt.write_count)
        self.assertEqual(nt[2], nt.read_bytes)
        self.assertEqual(nt[3], nt.write_bytes)
        if not (OPENBSD or NETBSD):
            self.assertEqual(nt[4], nt.read_time)
            self.assertEqual(nt[5], nt.write_time)
            if LINUX:
                self.assertEqual(nt[6], nt.read_merged_count)
                self.assertEqual(nt[7], nt.write_merged_count)
                self.assertEqual(nt[8], nt.busy_time)
            elif FREEBSD:
                self.assertEqual(nt[6], nt.busy_time)
        for name in nt._fields:
            assert getattr(nt, name) >= 0, nt
    ret = psutil.disk_io_counters(perdisk=False)
    assert ret is not None, 'no disks on this system?'
    check_ntuple(ret)
    ret = psutil.disk_io_counters(perdisk=True)
    self.assertEqual(len(ret), len(set(ret)))
    for key in ret:
        assert key, key
        check_ntuple(ret[key])