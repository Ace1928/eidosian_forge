import re
import copy
import inspect
import ast
import textwrap
def expand_varargs(expand_number):
    """
    Decorator to expand the variable length (starred) argument in a function
    signature with a fixed number of arguments.

    Parameters
    ----------
    expand_number: int
        The number of fixed arguments that should replace the variable length
        argument

    Returns
    -------
    function
        Decorator Function
    """
    if not isinstance(expand_number, int) or expand_number < 0:
        raise ValueError('expand_number must be a non-negative integer')

    def _expand_varargs(fn):
        fn_ast = function_to_ast(fn)
        fn_expanded_ast = expand_function_ast_varargs(fn_ast, expand_number)
        return function_ast_to_function(fn_expanded_ast, stacklevel=2)
    return _expand_varargs