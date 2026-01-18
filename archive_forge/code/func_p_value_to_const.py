import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_value_to_const(p):
    """
        value : QUOTED_STRING
              | NUMBER
              | TRUE
              | FALSE
              | NULL
        """
    p[0] = expressions.Constant(p[1])