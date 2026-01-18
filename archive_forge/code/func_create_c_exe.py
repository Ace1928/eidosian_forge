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
def create_c_exe(path, c_code=None):
    """Create a compiled C executable in the given location."""
    assert not os.path.exists(path), path
    if not which('gcc'):
        raise unittest.SkipTest('gcc is not installed')
    if c_code is None:
        c_code = textwrap.dedent('\n            #include <unistd.h>\n            int main() {\n                pause();\n                return 1;\n            }\n            ')
    else:
        assert isinstance(c_code, str), c_code
    atexit.register(safe_rmpath, path)
    with open(get_testfn(suffix='.c'), 'w') as f:
        f.write(c_code)
    try:
        subprocess.check_call(['gcc', f.name, '-o', path])
    finally:
        safe_rmpath(f.name)
    return path