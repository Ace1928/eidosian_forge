from docstring_parser.common import DocstringReturns
from docstring_parser.util import combine_docstrings
def fun2(arg_b, arg_c, arg_d, arg_e):
    """short_description: fun2

        long_description: fun2

        :param arg_b: fun2
        :param arg_c: fun2
        :param arg_e: fun2
        """
    assert arg_b and arg_c and arg_d and arg_e