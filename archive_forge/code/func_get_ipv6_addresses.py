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
def get_ipv6_addresses(ifname):
    with open('/proc/net/if_inet6') as f:
        all_fields = []
        for line in f.readlines():
            fields = line.split()
            if fields[-1] == ifname:
                all_fields.append(fields)
        if len(all_fields) == 0:
            raise ValueError('could not find interface %r' % ifname)
    for i in range(len(all_fields)):
        unformatted = all_fields[i][0]
        groups = []
        for j in range(0, len(unformatted), 4):
            groups.append(unformatted[j:j + 4])
        formatted = ':'.join(groups)
        packed = socket.inet_pton(socket.AF_INET6, formatted)
        all_fields[i] = socket.inet_ntop(socket.AF_INET6, packed)
    return all_fields