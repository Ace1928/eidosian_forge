from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_first_step_callback(self):
    if self.first_step_expr is None:
        return None
    return self._callback_factory([self.first_step_expr])