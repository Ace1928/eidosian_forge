import datetime
import errno
import os
import re
import subprocess
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import PYTHON_EXE
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
def ps_vsz(pid):
    field = 'vsz'
    if AIX:
        field = 'vsize'
    return ps(field, pid)