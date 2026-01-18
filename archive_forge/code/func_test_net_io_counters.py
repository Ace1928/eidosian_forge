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
@unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
def test_net_io_counters(self):

    def check_ntuple(nt):
        self.assertEqual(nt[0], nt.bytes_sent)
        self.assertEqual(nt[1], nt.bytes_recv)
        self.assertEqual(nt[2], nt.packets_sent)
        self.assertEqual(nt[3], nt.packets_recv)
        self.assertEqual(nt[4], nt.errin)
        self.assertEqual(nt[5], nt.errout)
        self.assertEqual(nt[6], nt.dropin)
        self.assertEqual(nt[7], nt.dropout)
        assert nt.bytes_sent >= 0, nt
        assert nt.bytes_recv >= 0, nt
        assert nt.packets_sent >= 0, nt
        assert nt.packets_recv >= 0, nt
        assert nt.errin >= 0, nt
        assert nt.errout >= 0, nt
        assert nt.dropin >= 0, nt
        assert nt.dropout >= 0, nt
    ret = psutil.net_io_counters(pernic=False)
    check_ntuple(ret)
    ret = psutil.net_io_counters(pernic=True)
    self.assertNotEqual(ret, [])
    for key in ret:
        self.assertTrue(key)
        self.assertIsInstance(key, str)
        check_ntuple(ret[key])