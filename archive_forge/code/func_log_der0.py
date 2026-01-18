from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def log_der0(*args):
    """
    Derivative of math.log() with respect to its first argument.

    Works whether 1 or 2 arguments are given.
    """
    if len(args) == 1:
        return 1 / args[0]
    else:
        return 1 / args[0] / math.log(args[1])