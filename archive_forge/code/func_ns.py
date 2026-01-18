import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
@property
def ns(self):
    """Number of substances"""
    return len(self.substances)