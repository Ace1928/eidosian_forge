import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def local_reset_stats(self):
    self.false_positives = 0
    self.false_negatives = 0
    self.true_positives = 0
    self.true_negatives = 0