from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def autonomous_post_processor(x, y, p):
    try:
        y[0][0, 0]
    except:
        pass
    else:
        return zip(*[autonomous_post_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
    return (y[..., -1], y[..., :-1], p)