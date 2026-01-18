from __future__ import (absolute_import, division, print_function)
import math
import os
from functools import reduce
from operator import mul
import numpy as np  # Lambdify requires numpy
import warnings
def prnt(self, e):
    return '%s(%s)' % (v, ', '.join((self._print(a) for a in e.args)))