from functools import partial
from sympy.strategies import chain, minimize
from sympy.strategies.core import identity
import sympy.strategies.branch as branch
from sympy.strategies.branch import yieldify
def allresults(tree, leaf=yieldify):
    """ Execute a strategic tree.  Return all possibilities.

    Returns a lazy iterator of all possible results

    Exhaustiveness
    --------------

    This is an exhaustive algorithm.  In the example

        ([a, b], [c, d])

    All of the results from

        (a, c), (b, c), (a, d), (b, d)

    are returned.  This can lead to combinatorial blowup.

    See sympy.strategies.greedy for details on input
    """
    return treeapply(tree, {list: branch.multiplex, tuple: branch.chain}, leaf=leaf)