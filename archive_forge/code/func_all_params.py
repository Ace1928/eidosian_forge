import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def all_params(self, variables, backend=math):
    return [v(variables, backend=backend) if isinstance(v, Expr) else v for v in [variables[k] for k in self.parameter_keys]]