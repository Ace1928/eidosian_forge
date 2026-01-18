import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
def check_constant_unicode_cache():
    pyfunc = constant_unicode_cache
    cfunc = njit(cache=True)(pyfunc)
    exp_hv, exp_str = pyfunc()
    got_hv, got_str = cfunc()
    assert exp_hv == got_hv
    assert exp_str == got_str