import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def set_powerlimits(self, lims):
    """
        Set size thresholds for scientific notation.

        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \\times 10^\\mathrm{exp}` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.

            The default limits are controlled by :rc:`axes.formatter.limits`.

            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.

            Typically, *min_exp* will be negative and *max_exp* will be
            positive.

            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \\times 10^{-3}, 9.9 \\times 10^{-3}, 0.01,`
            :math:`9999, 1 \\times 10^4`.

        See Also
        --------
        ScalarFormatter.set_scientific
        """
    if len(lims) != 2:
        raise ValueError("'lims' must be a sequence of length 2")
    self._powerlimits = lims