import datetime
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import warnings
import psutil
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import super
from psutil._compat import which
from psutil.tests import APPVEYOR
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_BATTERY
from psutil.tests import IS_64BIT
from psutil.tests import PY3
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
def powershell(cmd):
    """Currently not used, but available just in case. Usage:

    >>> powershell(
        "Get-CIMInstance Win32_PageFileUsage | Select AllocatedBaseSize")
    """
    if not which('powershell.exe'):
        raise unittest.SkipTest('powershell.exe not available')
    cmdline = 'powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive ' + '-NoProfile -WindowStyle Hidden -Command "%s"' % cmd
    return sh(cmdline)