from itertools import product
from sympy.strategies.util import basic_fns
from .core import chain, identity, do_one
def sall(brule, fns=basic_fns):
    """ Strategic all - apply rule to args """
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            yield expr
        else:
            myop = op(expr)
            argss = product(*map(brule, children(expr)))
            for args in argss:
                yield new(myop, *args)
    return all_rl