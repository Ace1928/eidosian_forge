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
def assertNoNRTLeak(self):
    """
        A context manager that asserts no NRT leak was created during
        the execution of the enclosed block.
        """
    old = rtsys.get_allocation_stats()
    yield
    new = rtsys.get_allocation_stats()
    total_alloc = new.alloc - old.alloc
    total_free = new.free - old.free
    total_mi_alloc = new.mi_alloc - old.mi_alloc
    total_mi_free = new.mi_free - old.mi_free
    self.assertEqual(total_alloc, total_free, 'number of data allocs != number of data frees')
    self.assertEqual(total_mi_alloc, total_mi_free, 'number of meminfo allocs != number of meminfo frees')