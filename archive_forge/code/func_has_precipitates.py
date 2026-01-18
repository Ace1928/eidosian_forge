from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
def has_precipitates(self, substances):
    for s_name in chain(self.reac.keys(), self.prod.keys(), self.inact_reac.keys(), self.inact_prod.keys()):
        if getattr(substances[s_name], 'phase_idx', 0) > 0:
            return True
    return False