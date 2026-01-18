import functools
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import signature
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from .._config import config_context, get_config
from .validation import _is_arraylike_not_scalar
def generate_invalid_param_val(constraint):
    """Return a value that does not satisfy the constraint.

    Raises a NotImplementedError if there exists no invalid value for this constraint.

    This is only useful for testing purpose.

    Parameters
    ----------
    constraint : _Constraint instance
        The constraint to generate a value for.

    Returns
    -------
    val : object
        A value that does not satisfy the constraint.
    """
    if isinstance(constraint, StrOptions):
        return f'not {' or '.join(constraint.options)}'
    if isinstance(constraint, MissingValues):
        return np.array([1, 2, 3])
    if isinstance(constraint, _VerboseHelper):
        return -1
    if isinstance(constraint, HasMethods):
        return type('HasNotMethods', (), {})()
    if isinstance(constraint, _IterablesNotString):
        return 'a string'
    if isinstance(constraint, _CVObjects):
        return 'not a cv object'
    if isinstance(constraint, Interval) and constraint.type is Integral:
        if constraint.left is not None:
            return constraint.left - 1
        if constraint.right is not None:
            return constraint.right + 1
        raise NotImplementedError
    if isinstance(constraint, Interval) and constraint.type in (Real, RealNotInt):
        if constraint.left is not None:
            return constraint.left - 1e-06
        if constraint.right is not None:
            return constraint.right + 1e-06
        if constraint.closed in ('right', 'neither'):
            return -np.inf
        if constraint.closed in ('left', 'neither'):
            return np.inf
        return np.nan
    raise NotImplementedError