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
def as_reactions(self, kf=None, kb=None, units=None, variables=None, backend=math, new_name=None, **kwargs):
    """Creates a forward and backward :class:`Reaction` pair

        Parameters
        ----------
        kf : float or RateExpr
        kb : float or RateExpr
        units : module
        variables : dict, optional
        backend : module

        """
    nb = sum(self.prod.values())
    nf = sum(self.reac.values())
    if units is None:
        if hasattr(kf, 'units') or hasattr(kb, 'units'):
            raise ValueError('units missing')
        c0 = 1
    else:
        c0 = 1 * units.molar
    if kf is None:
        fw_name = self.name
        bw_name = new_name
        if kb is None:
            try:
                kf, kb = self.param
            except TypeError:
                raise ValueError('Exactly one rate needs to be provided')
        else:
            kf = kb * self.param * c0 ** (nb - nf)
    elif kb is None:
        kb = kf / (self.param * c0 ** (nb - nf))
        fw_name = new_name
        bw_name = self.name
    else:
        raise ValueError('Exactly one rate needs to be provided')
    return (Reaction(self.reac, self.prod, kf, self.inact_reac, self.inact_prod, ref=self.ref, name=fw_name, **kwargs), Reaction(self.prod, self.reac, kb, self.inact_prod, self.inact_reac, ref=self.ref, name=bw_name, **kwargs))