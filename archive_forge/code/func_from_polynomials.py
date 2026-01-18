from functools import reduce
import math
from operator import add
from ._expr import Expr
@classmethod
def from_polynomials(cls, bounds, polys, inject=[], **kwargs):
    if any((p.parameter_keys != (parameter,) for p in polys)):
        raise ValueError('Mixed parameter_keys')
    npolys = len(polys)
    if len(bounds) != npolys:
        raise ValueError('Length mismatch')
    meta = reduce(add, [[len(p.args[p.skip_poly:]) - 1, l, u] for (l, u), p in zip(bounds, polys)])
    p_args = reduce(add, [p.args[p.skip_poly:] for p in polys])
    return cls(inject + [npolys] + meta + p_args, **kwargs)