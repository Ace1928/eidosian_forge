import enum
import errno
import multiprocessing
import os
import stat
import time
import traceback
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import long
from psutil._compat import unicode
from psutil.tests import CI_TESTING
from psutil.tests import VALID_PROC_STATUSES
from psutil.tests import PsutilTestCase
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import is_namedtuple
from psutil.tests import is_win_secure_system_proc
from psutil.tests import process_namespace
from psutil.tests import serialrun
def nice(self, ret, info):
    self.assertIsInstance(ret, int)
    if POSIX:
        assert -20 <= ret <= 20, ret
    else:
        priorities = [getattr(psutil, x) for x in dir(psutil) if x.endswith('_PRIORITY_CLASS')]
        self.assertIn(ret, priorities)
        if PY3:
            self.assertIsInstance(ret, enum.IntEnum)
        else:
            self.assertIsInstance(ret, int)