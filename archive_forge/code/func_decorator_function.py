from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
@wraps(function)
def decorator_function(*args, **kwargs):
    """
        Decorator function to suppress warnings.

        Parameters
        ----------
        args : arguments, optional
            Arguments passed to function to be decorated.
        kwargs : keyword arguments, optional
            Keyword arguments passed to function to be decorated.

        Returns
        -------
        decorated function
            Decorated function.

        """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)