import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
def p_value_call(this, p):
    """
            func : value '(' args ')'
            """
    arg = ()
    if len(p) > 4:
        arg = p[3]
    p[0] = expressions.Function('#call', p[1], *arg)