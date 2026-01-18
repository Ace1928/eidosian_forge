from functools import reduce
from sympy.core import Basic, S, Mul, PoleError, expand_mul
from sympy.core.cache import cacheit
from sympy.core.numbers import ilcm, I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up
from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis
def build_expression_tree(Omega, rewrites):
    """ Helper function for rewrite.

    We need to sort Omega (mrv set) so that we replace an expression before
    we replace any expression in terms of which it has to be rewritten::

        e1 ---> e2 ---> e3
                 \\
                  -> e4

    Here we can do e1, e2, e3, e4 or e1, e2, e4, e3.
    To do this we assemble the nodes into a tree, and sort them by height.

    This function builds the tree, rewrites then sorts the nodes.
    """

    class Node:

        def __init__(self):
            self.before = []
            self.expr = None
            self.var = None

        def ht(self):
            return reduce(lambda x, y: x + y, [x.ht() for x in self.before], 1)
    nodes = {}
    for expr, v in Omega:
        n = Node()
        n.var = v
        n.expr = expr
        nodes[v] = n
    for _, v in Omega:
        if v in rewrites:
            n = nodes[v]
            r = rewrites[v]
            for _, v2 in Omega:
                if r.has(v2):
                    n.before.append(nodes[v2])
    return nodes