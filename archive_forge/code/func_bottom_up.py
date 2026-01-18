from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one
def bottom_up(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the bottom nodes first."""
    return chain(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)