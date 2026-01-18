import itertools
import math
import operator
import random
from functools import reduce
def effectiveness(self):
    """-> returns the average effectiveness of this library set"""
    sum = 0.0
    for rg in self.rgroups:
        sum += rg.effectiveness()
    return sum / len(self.rgroups)