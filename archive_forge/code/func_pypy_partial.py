import sys
import platform
def pypy_partial(val):
    """
    Adjust for variable stacklevel on partial under PyPy.

    Workaround for #327.
    """
    is_pypy = platform.python_implementation() == 'PyPy'
    return val + is_pypy