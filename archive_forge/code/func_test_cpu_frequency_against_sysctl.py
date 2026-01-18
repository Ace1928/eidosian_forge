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
def test_cpu_frequency_against_sysctl(self):
    sensor = 'dev.cpu.0.freq'
    try:
        sysctl_result = int(sysctl(sensor))
    except RuntimeError:
        self.skipTest('frequencies not supported by kernel')
    self.assertEqual(psutil.cpu_freq().current, sysctl_result)
    sensor = 'dev.cpu.0.freq_levels'
    sysctl_result = sysctl(sensor)
    max_freq = int(sysctl_result.split()[0].split('/')[0])
    min_freq = int(sysctl_result.split()[-1].split('/')[0])
    self.assertEqual(psutil.cpu_freq().max, max_freq)
    self.assertEqual(psutil.cpu_freq().min, min_freq)