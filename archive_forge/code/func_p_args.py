import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_args(p):
    """
        args : arglist
             | named_arglist
             | arglist ',' named_arglist
             | incomplete_arglist ',' named_arglist
             |
        """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1] + p[3]