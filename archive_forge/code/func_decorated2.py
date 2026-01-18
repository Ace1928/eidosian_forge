from docstring_parser.common import DocstringReturns
from docstring_parser.util import combine_docstrings
@combine_docstrings(fun1, fun2, exclude=[DocstringReturns])
def decorated2(arg_a, arg_b, arg_c, arg_d, arg_e, arg_f):
    assert arg_a and arg_b and arg_c and arg_d and arg_e and arg_f