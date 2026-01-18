import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def new_substances(coll):
    return OrderedDict([(k, v) for k, v in self.substances.items() if any([k in r.keys() for r in coll])])