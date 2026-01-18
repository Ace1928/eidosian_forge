import platform
import signal
import unittest
import psutil
from psutil import AIX
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import long
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYPY
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import create_sockets
from psutil.tests import enum
from psutil.tests import is_namedtuple
from psutil.tests import kernel_version
def test_linux_ioprio_windows(self):
    ae = self.assertEqual
    ae(hasattr(psutil, 'IOPRIO_HIGH'), WINDOWS)
    ae(hasattr(psutil, 'IOPRIO_NORMAL'), WINDOWS)
    ae(hasattr(psutil, 'IOPRIO_LOW'), WINDOWS)
    ae(hasattr(psutil, 'IOPRIO_VERYLOW'), WINDOWS)