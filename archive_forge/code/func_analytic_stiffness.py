from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def analytic_stiffness(self, xyp=None):
    """ Running stiffness ratio from last integration.

        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the (analytic) jacobian matrix.

        See :meth:`ODESys.stiffness` for more info.
        """
    return self.stiffness(xyp, self._get_analytic_stiffness_cb())