import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_arg_list(p):
    """
        arglist : value
                | ',' arglist
                | arglist ',' arglist
                | incomplete_arglist ',' arglist
        """
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [utils.NO_VALUE] + p[2]
    elif len(p) == 4:
        p[0] = p[1] + p[3]