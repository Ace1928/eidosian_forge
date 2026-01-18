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
def make_dummy_type(self):
    """Use to generate a dummy type unique to this test. Returns a python
        Dummy class and a corresponding Numba type DummyType."""
    test_id = self.id()
    DummyType = type('DummyTypeFor{}'.format(test_id), (types.Opaque,), {})
    dummy_type = DummyType('my_dummy')
    register_model(DummyType)(OpaqueModel)

    class Dummy(object):
        pass

    @typeof_impl.register(Dummy)
    def typeof_dummy(val, c):
        return dummy_type

    @unbox(DummyType)
    def unbox_dummy(typ, obj, c):
        return NativeValue(c.context.get_dummy_value())
    return (Dummy, DummyType)