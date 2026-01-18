import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def ranges0(*args):
    return [qfun(args[1], args[0]) if callable(qfun) else qfun, rfun(args[1], args[0]) if callable(rfun) else rfun]