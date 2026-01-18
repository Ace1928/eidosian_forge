import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_val_to_function(p):
    """
        value : func
        """
    p[0] = p[1]