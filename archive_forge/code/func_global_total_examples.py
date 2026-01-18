import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def global_total_examples(self):
    return self.global_false_negatives + self.global_false_positives + self.global_true_negatives + self.global_true_positives