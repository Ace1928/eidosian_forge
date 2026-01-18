from itertools import product
from sympy.strategies.util import basic_fns
from .core import chain, identity, do_one
def all_rl(expr):
    if leaf(expr):
        yield expr
    else:
        myop = op(expr)
        argss = product(*map(brule, children(expr)))
        for args in argss:
            yield new(myop, *args)