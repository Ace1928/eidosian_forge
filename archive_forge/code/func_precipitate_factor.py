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
def precipitate_factor(self, substances, sc_concs):
    factor = 1
    for r, n in self.reac.items():
        if r.precipitate:
            factor *= sc_concs[substances.index(r)] ** (-n)
    for p, n in self.prod.items():
        if p.precipitate:
            factor *= sc_concs[substances.index(p)] ** n
    return factor