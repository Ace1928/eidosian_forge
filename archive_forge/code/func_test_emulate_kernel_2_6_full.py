from __future__ import division
import collections
import contextlib
import errno
import glob
import io
import os
import re
import shutil
import socket
import struct
import textwrap
import time
import unittest
import warnings
import psutil
from psutil import LINUX
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import basestring
from psutil._compat import u
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_RLIMIT
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import retry_on_failure
from psutil.tests import safe_rmpath
from psutil.tests import sh
from psutil.tests import skip_on_not_implemented
from psutil.tests import which
def test_emulate_kernel_2_6_full(self):
    content = '   3    0   hda 1 2 3 4 5 6 7 8 9 10 11'
    with mock_open_content({'/proc/diskstats': content}):
        with mock.patch('psutil._pslinux.is_storage_device', return_value=True):
            ret = psutil.disk_io_counters(nowrap=False)
            self.assertEqual(ret.read_count, 1)
            self.assertEqual(ret.read_merged_count, 2)
            self.assertEqual(ret.read_bytes, 3 * SECTOR_SIZE)
            self.assertEqual(ret.read_time, 4)
            self.assertEqual(ret.write_count, 5)
            self.assertEqual(ret.write_merged_count, 6)
            self.assertEqual(ret.write_bytes, 7 * SECTOR_SIZE)
            self.assertEqual(ret.write_time, 8)
            self.assertEqual(ret.busy_time, 10)