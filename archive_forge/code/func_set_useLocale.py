import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def set_useLocale(self, val):
    """
        Set whether to use locale settings for decimal sign and positive sign.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
    if val is None:
        self._useLocale = mpl.rcParams['axes.formatter.use_locale']
    else:
        self._useLocale = val