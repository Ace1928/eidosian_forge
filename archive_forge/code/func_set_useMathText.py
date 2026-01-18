import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def set_useMathText(self, val):
    if val is None:
        self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']
    else:
        self._useMathText = val