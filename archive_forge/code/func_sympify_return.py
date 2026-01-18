from functools import wraps
from .sympify import SympifyError, sympify
def sympify_return(*args):
    """Function/method decorator to sympify arguments automatically

    See the docstring of sympify_method_args for explanation.
    """

    def wrapper(func):
        return _SympifyWrapper(func, args)
    return wrapper