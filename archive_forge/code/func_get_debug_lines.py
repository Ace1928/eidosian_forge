from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def get_debug_lines(fn):
    metadata = self._get_metadata(fn, fn.signatures[0])
    lines = self._get_lines_from_debuginfo(metadata)
    return lines