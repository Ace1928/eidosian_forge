import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def set_one_half(self, one_half):
    """
        Set the way one half is displayed.

        one_half : str, default: r"\\frac{1}{2}"
            The string used to represent 1/2.
        """
    self._one_half = one_half