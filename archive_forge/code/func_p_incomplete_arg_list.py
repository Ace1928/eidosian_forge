import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_incomplete_arg_list(p):
    """
        incomplete_arglist : arglist ','
        """
    p[0] = p[1] + [utils.NO_VALUE]