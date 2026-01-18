import itertools
import math
import operator
import random
from functools import reduce
def isValid(self):
    """If we have an empty set for any rgroup, return False"""
    for rg in self.rgroups:
        if len(rg.sidechains) == 0:
            return False
    return True