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
@staticmethod
def eliminate(rxns, wrt):
    """Linear combination coefficients for elimination of a substance

        Parameters
        ----------
        rxns : iterable of Equilibrium instances
        wrt : str (substance key)

        Examples
        --------
        >>> e1 = Equilibrium({'Cd+2': 4, 'H2O': 4}, {'Cd4(OH)4+4': 1, 'H+': 4}, 10**-32.5)
        >>> e2 = Equilibrium({'Cd(OH)2(s)': 1}, {'Cd+2': 1, 'OH-': 2}, 10**-14.4)
        >>> Equilibrium.eliminate([e1, e2], 'Cd+2')
        [1, 4]
        >>> print(1*e1 + 4*e2)
        4 Cd(OH)2(s) + 4 H2O = Cd4(OH)4+4 + 4 H+ + 8 OH-; 7.94e-91

        """
    import sympy
    viol = [r.net_stoich([wrt])[0] for r in rxns]
    factors = defaultdict(int)
    for v in viol:
        for f in sympy.primefactors(v):
            factors[f] = max(factors[f], sympy.Abs(v // f))
    rcd = reduce(mul, (k ** v for k, v in factors.items()))
    viol[0] *= -1
    return [rcd // v for v in viol]