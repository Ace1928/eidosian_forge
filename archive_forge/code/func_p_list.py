import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_list(p):
    """
        value : INDEXER args ']' %prec LIST
        """
    p[0] = expressions.ListExpression(*p[2])