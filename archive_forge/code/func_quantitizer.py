import inspect
import os
import re
import string
import sys
import warnings
from functools import partial, wraps
def quantitizer(base_function, handler_function=lambda *args, **kwargs: 1.0):
    """
    wraps a function so that it works properly with physical quantities
    (Quantities).
    arguments:
        base_function - the function to be wrapped
        handler_function - a function which takes the same arguments as the
            base_function  and returns a Quantity (or tuple of Quantities)
            which has (have) the units that the output of base_function should
            have.
        returns:
            a wrapped version of base_function that takes the same arguments
            and works with physical quantities. It will have almost the same
            __name__ and almost the same __doc__.
    """
    from .quantity import Quantity

    def wrapped_function(*args, **kwargs):
        handler_quantities = handler_function(*args, **kwargs)
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], Quantity):
                args[i] = args[i].simplified
                args[i] = args[i].magnitude
        args = tuple(args)
        for i in kwargs:
            if isinstance(kwargs[i], Quantity):
                kwargs[i] = kwargs[i].simplifed()
                kwargs[i] = kwargs[i].magnitude
        result = base_function(*args, **kwargs)
        result = list(result)
        length = min(len(handler_quantities), len(result))
        for i in range(length):
            if isinstance(handler_quantities[i], Quantity):
                result[i] = Quantity(result[i], handler_quantities[i].dimensionality.simplified)
                result[i] = result[i].rescale(handler_quantities[i].dimensionality)
        result = tuple(result)
        return result
    wrapped_function.__name__ = base_function.__name__ + '_QWrap'
    wrapped_function.__doc__ = 'this function has been wrapped to work with Quantities\n' + base_function.__doc__
    return wrapped_function