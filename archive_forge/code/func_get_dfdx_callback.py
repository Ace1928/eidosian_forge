from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_dfdx_callback(self):
    """ Generate a callback for evaluating derivative of ``self.exprs`` """
    dfdx_exprs = self.get_dfdx()
    if dfdx_exprs is False:
        return None
    return self._callback_factory(dfdx_exprs)