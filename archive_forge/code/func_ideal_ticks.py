import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def ideal_ticks(x):
    return 10 ** x if x < 0 else 1 - 10 ** (-x) if x > 0 else 0.5