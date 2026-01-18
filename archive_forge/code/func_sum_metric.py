import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def sum_metric(self):
    return self._calc_mcc(self.lcm) * self.num_inst