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
def glob_mock(path):
    if path == '/sys/class/hwmon/hwmon*/temp*_*':
        return []
    elif path == '/sys/class/hwmon/hwmon*/device/temp*_*':
        return []
    elif path == '/sys/class/thermal/thermal_zone*':
        return ['/sys/class/thermal/thermal_zone0']
    elif path == '/sys/class/thermal/thermal_zone0/trip_point*':
        return ['/sys/class/thermal/thermal_zone1/trip_point_0_type', '/sys/class/thermal/thermal_zone1/trip_point_0_temp']
    return []