import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def sort_substances_inplace(self, key=lambda kv: kv[0]):
    """Sorts the OrderedDict attribute ``substances``"""
    self.substances = OrderedDict(sorted(self.substances.items(), key=key))