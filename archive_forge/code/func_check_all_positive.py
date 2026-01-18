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
def check_all_positive(self, throw=False):
    """Checks if all stoichiometric coefficients are positive"""
    for nam, cont in [(nam, getattr(self, nam)) for nam in 'reac prod inact_reac inact_prod'.split()]:
        for k, v in cont.items():
            if v < 0:
                if throw:
                    raise ValueError('Found a negative stoichiometry for %s in %s.' % (k, nam))
                else:
                    return False
    return True