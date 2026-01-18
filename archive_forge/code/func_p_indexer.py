import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_indexer(p):
    """
        value : value INDEXER args ']'
        """
    p[0] = expressions.IndexExpression(p[1], *p[3])