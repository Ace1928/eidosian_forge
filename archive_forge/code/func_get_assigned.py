import functools
import inspect
import warnings
def get_assigned(decorator):
    """Helper to fix/workaround https://bugs.python.org/issue3445"""
    return functools.WRAPPER_ASSIGNMENTS