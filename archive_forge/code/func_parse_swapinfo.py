import datetime
import os
import re
import time
import unittest
import psutil
from psutil import BSD
from psutil import FREEBSD
from psutil import NETBSD
from psutil import OPENBSD
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
@staticmethod
def parse_swapinfo():
    output = sh('swapinfo -k').splitlines()[-1]
    parts = re.split('\\s+', output)
    if not parts:
        raise ValueError("Can't parse swapinfo: %s" % output)
    total, used, free = (int(p) * 1024 for p in parts[1:4])
    return (total, used, free)