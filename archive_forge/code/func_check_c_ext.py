import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
@contextlib.contextmanager
def check_c_ext(self, extdir, name):
    sys.path.append(extdir)
    try:
        lib = import_dynamic(name)
        yield lib
    finally:
        sys.path.remove(extdir)
        sys.modules.pop(name, None)