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
def parse_meminfo(look_for):
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(look_for):
                return int(line.split()[1]) * 1024
    raise ValueError("can't find %s" % look_for)