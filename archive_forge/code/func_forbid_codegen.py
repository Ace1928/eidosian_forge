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
def forbid_codegen():
    """
    Forbid LLVM code generation during the execution of the context
    manager's enclosed block.

    If code generation is invoked, a RuntimeError is raised.
    """
    from numba.core import codegen
    patchpoints = ['CPUCodeLibrary._finalize_final_module']
    old = {}

    def fail(*args, **kwargs):
        raise RuntimeError('codegen forbidden by test case')
    try:
        for name in patchpoints:
            parts = name.split('.')
            obj = codegen
            for attrname in parts[:-1]:
                obj = getattr(obj, attrname)
            attrname = parts[-1]
            value = getattr(obj, attrname)
            assert callable(value), '%r should be callable' % name
            old[obj, attrname] = value
            setattr(obj, attrname, fail)
        yield
    finally:
        for (obj, attrname), value in old.items():
            setattr(obj, attrname, value)