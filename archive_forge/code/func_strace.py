import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
def strace(work, syscalls, timeout=10):
    """Runs strace whilst executing the function work() in the current process,
    captures the listed syscalls (list of strings). Takes an optional timeout in
    seconds, default is 10, if this is exceeded the process will be sent a
    SIGKILL. Returns a list of lines that are output by strace.
    """
    with tempfile.NamedTemporaryFile('w+t') as ntf:
        parent_pid = os.getpid()
        strace_binary = shutil.which('strace')
        if strace_binary is None:
            raise ValueError("No valid 'strace' binary could be found")
        cmd = [strace_binary, '-q', '-p', str(parent_pid), '-e', ','.join(syscalls), '-o', ntf.name]
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        strace_pid = popen.pid
        thread_timeout = threading.Timer(timeout, popen.kill)
        thread_timeout.start()

        def check_return(problem=''):
            ret = popen.returncode
            if ret != 0:
                msg = f'strace exited non-zero, process return code was:{ret}. {problem}'
                raise RuntimeError(msg)
        try:
            thread_comms = threading.Thread(target=popen.communicate)
            thread_comms.start()
            work()
            ntf.flush()
            if popen.poll() is None:
                os.kill(strace_pid, signal.SIGINT)
            else:
                problem = 'If this is SIGKILL, increase the timeout?'
                check_return(problem)
            popen.wait()
            check_return()
            strace_data = ntf.readlines()
        finally:
            thread_comms.join()
            thread_timeout.cancel()
    return strace_data