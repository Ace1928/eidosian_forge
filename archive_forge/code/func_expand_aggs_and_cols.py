from __future__ import annotations
from packaging.version import Version
import inspect
import warnings
import os
from math import isnan
import numpy as np
import pandas as pd
import xarray as xr
from datashader.utils import Expr, ngjit
from datashader.macros import expand_varargs
def expand_aggs_and_cols(self, append):
    """
        Create a decorator that can be used on functions that accept
        *aggs_and_cols as a variable length argument. The decorator will
        replace *aggs_and_cols with a fixed number of arguments.

        The appropriate fixed number of arguments is calculated from the input
        append function.

        Rationale: When we know the fixed length of a variable length
        argument, replacing it with fixed arguments can help numba better
        optimize the the function.

        If this ever causes problems in the future, this decorator can be
        safely removed without changing the functionality of the decorated
        function.

        Parameters
        ----------
        append: function
            The append function for the current aggregator

        Returns
        -------
        function
            Decorator function
        """
    return self._expand_aggs_and_cols(append, self.ndims, self.antialiased)