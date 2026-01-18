import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def as_substance_index(self, substance_key):
    """Returns the index of a Substance in the system"""
    if isinstance(substance_key, int):
        return substance_key
    else:
        return list(self.substances.keys()).index(substance_key)