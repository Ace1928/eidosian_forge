from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one
def bottom_up_once(rule, fns=basic_fns):
    """Apply a rule up a tree - stop on success."""
    return do_one(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)