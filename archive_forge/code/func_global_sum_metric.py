import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def global_sum_metric(self):
    return self._calc_mcc(self.gcm) * self.global_num_inst