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
@contextlib.contextmanager
def redirect_fd(fd):
    """
    Temporarily redirect *fd* to a pipe's write end and return a file object
    wrapping the pipe's read end.
    """
    from numba import _helperlib
    libnumba = ctypes.CDLL(_helperlib.__file__)
    libnumba._numba_flush_stdout()
    save = os.dup(fd)
    r, w = os.pipe()
    try:
        os.dup2(w, fd)
        yield io.open(r, 'r')
    finally:
        libnumba._numba_flush_stdout()
        os.close(w)
        os.dup2(save, fd)
        os.close(save)