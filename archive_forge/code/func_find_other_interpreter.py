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
@staticmethod
def find_other_interpreter():
    code = 'import sys; sys.stdout.write(str(sys.maxsize > 2**32))'
    for filename in glob.glob('C:\\Python*\\python.exe'):
        proc = subprocess.Popen(args=[filename, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = proc.communicate()
        proc.wait()
        if output == str(not IS_64BIT):
            return filename