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
def test_avail_old_missing_fields(self):
    content = textwrap.dedent('            Active:          9444728 kB\n            Active(anon):    6145416 kB\n            Buffers:          287952 kB\n            Cached:          4818144 kB\n            Inactive(file):  1578132 kB\n            Inactive(anon):   574764 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            Shmem:            577588 kB\n            ').encode()
    with mock_open_content({'/proc/meminfo': content}) as m:
        with warnings.catch_warnings(record=True) as ws:
            ret = psutil.virtual_memory()
        assert m.called
        self.assertEqual(ret.available, 2057400 * 1024 + 4818144 * 1024)
        w = ws[0]
        self.assertIn("inactive memory stats couldn't be determined", str(w.message))