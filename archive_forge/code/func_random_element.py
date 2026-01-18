from .sage_helper import _within_sage
from .pari import *
import re
def random_element(self, min=-1, max=1):
    min = self(min)
    max = self(max)
    limit = (max - min) * self(2) ** self._precision
    normalizer = self(2.0) ** (-self._precision)
    return min + normalizer * gen.random(limit.gen)