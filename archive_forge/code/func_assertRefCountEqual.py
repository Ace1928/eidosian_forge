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
def assertRefCountEqual(self, *objects):
    gc.collect()
    rc = [sys.getrefcount(x) for x in objects]
    rc_0 = rc[0]
    for i in range(len(objects))[1:]:
        rc_i = rc[i]
        if rc_0 != rc_i:
            self.fail(f'Refcount for objects does not match. #0({rc_0}) != #{i}({rc_i}) does not match.')