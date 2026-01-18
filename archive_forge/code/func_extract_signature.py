import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def extract_signature(func, ignore_first=False):
    """Extract the function signature from the function.

    Args:
        func: The function whose signature should be extracted.
        ignore_first: True if the first argument should be ignored. This should
            be used when func is a method of a class.

    Returns:
        List of Parameter objects representing the function signature.
    """
    signature_parameters = list(get_signature(func).parameters.values())
    if ignore_first:
        if len(signature_parameters) == 0:
            raise ValueError(f"Methods must take a 'self' argument, but the method '{func.__name__}' does not have one.")
        signature_parameters = signature_parameters[1:]
    return signature_parameters