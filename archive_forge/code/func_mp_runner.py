import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
@classmethod
def mp_runner(cls, testname, outqueue):
    method = getattr(cls, testname)
    try:
        ok, msg = method()
    except Exception:
        msg = traceback.format_exc()
        ok = False
    outqueue.put({'status': ok, 'msg': msg})