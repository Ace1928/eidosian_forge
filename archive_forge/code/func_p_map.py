import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_map(p):
    """
        value : MAP args '}'
        """
    p[0] = expressions.MapExpression(*p[2])