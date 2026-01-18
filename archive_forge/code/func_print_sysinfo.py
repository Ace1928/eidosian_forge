from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
def print_sysinfo():
    import collections
    import datetime
    import getpass
    import locale
    import pprint
    try:
        import pip
    except ImportError:
        pip = None
    try:
        import wheel
    except ImportError:
        wheel = None
    info = collections.OrderedDict()
    if psutil.LINUX and which('lsb_release'):
        info['OS'] = sh('lsb_release -d -s')
    elif psutil.OSX:
        info['OS'] = 'Darwin %s' % platform.mac_ver()[0]
    elif psutil.WINDOWS:
        info['OS'] = 'Windows ' + ' '.join(map(str, platform.win32_ver()))
        if hasattr(platform, 'win32_edition'):
            info['OS'] += ', ' + platform.win32_edition()
    else:
        info['OS'] = '%s %s' % (platform.system(), platform.version())
    info['arch'] = ', '.join(list(platform.architecture()) + [platform.machine()])
    if psutil.POSIX:
        info['kernel'] = platform.uname()[2]
    info['python'] = ', '.join([platform.python_implementation(), platform.python_version(), platform.python_compiler()])
    info['pip'] = getattr(pip, '__version__', 'not installed')
    if wheel is not None:
        info['pip'] += ' (wheel=%s)' % wheel.__version__
    if psutil.POSIX:
        if which('gcc'):
            out = sh(['gcc', '--version'])
            info['gcc'] = str(out).split('\n')[0]
        else:
            info['gcc'] = 'not installed'
        s = platform.libc_ver()[1]
        if s:
            info['glibc'] = s
    info['fs-encoding'] = sys.getfilesystemencoding()
    lang = locale.getlocale()
    info['lang'] = '%s, %s' % (lang[0], lang[1])
    info['boot-time'] = datetime.datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
    info['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info['user'] = getpass.getuser()
    info['home'] = os.path.expanduser('~')
    info['cwd'] = os.getcwd()
    info['pyexe'] = PYTHON_EXE
    info['hostname'] = platform.node()
    info['PID'] = os.getpid()
    info['cpus'] = psutil.cpu_count()
    info['loadavg'] = '%.1f%%, %.1f%%, %.1f%%' % tuple([x / psutil.cpu_count() * 100 for x in psutil.getloadavg()])
    mem = psutil.virtual_memory()
    info['memory'] = '%s%%, used=%s, total=%s' % (int(mem.percent), bytes2human(mem.used), bytes2human(mem.total))
    swap = psutil.swap_memory()
    info['swap'] = '%s%%, used=%s, total=%s' % (int(swap.percent), bytes2human(swap.used), bytes2human(swap.total))
    info['pids'] = len(psutil.pids())
    pinfo = psutil.Process().as_dict()
    pinfo.pop('memory_maps', None)
    info['proc'] = pprint.pformat(pinfo)
    print('=' * 70, file=sys.stderr)
    for k, v in info.items():
        print('%-17s %s' % (k + ':', v), file=sys.stderr)
    print('=' * 70, file=sys.stderr)
    sys.stdout.flush()
    if WINDOWS:
        os.system('tasklist')
    elif which('ps'):
        os.system('ps aux')
    print('=' * 70, file=sys.stderr)
    sys.stdout.flush()