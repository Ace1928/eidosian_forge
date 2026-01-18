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
@njit(debug=True)
def sink(*x):
    pass