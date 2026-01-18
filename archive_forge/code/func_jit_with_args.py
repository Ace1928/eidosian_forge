import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def jit_with_args(name, argstring):
    code = 'def func(%(argstring)s):\n        return %(name)s(%(argstring)s)\n' % locals()
    pyfunc = compile_function('func', code, globals())
    return jit(nopython=True)(pyfunc)