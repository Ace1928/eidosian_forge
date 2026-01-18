from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def jacobian_singular(self):
    """ Returns True if Jacobian is singular, else False. """
    cses, (jac_in_cses,) = self.be.cse(self.get_jac())
    if jac_in_cses.nullspace():
        return True
    else:
        return False