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
def make_constraint(constraint):
    """Convert the constraint into the appropriate Constraint object.

    Parameters
    ----------
    constraint : object
        The constraint to convert.

    Returns
    -------
    constraint : instance of _Constraint
        The converted constraint.
    """
    if isinstance(constraint, str) and constraint == 'array-like':
        return _ArrayLikes()
    if isinstance(constraint, str) and constraint == 'sparse matrix':
        return _SparseMatrices()
    if isinstance(constraint, str) and constraint == 'random_state':
        return _RandomStates()
    if constraint is callable:
        return _Callables()
    if constraint is None:
        return _NoneConstraint()
    if isinstance(constraint, type):
        return _InstancesOf(constraint)
    if isinstance(constraint, (Interval, StrOptions, Options, HasMethods, MissingValues)):
        return constraint
    if isinstance(constraint, str) and constraint == 'boolean':
        return _Booleans()
    if isinstance(constraint, str) and constraint == 'verbose':
        return _VerboseHelper()
    if isinstance(constraint, str) and constraint == 'cv_object':
        return _CVObjects()
    if isinstance(constraint, Hidden):
        constraint = make_constraint(constraint.constraint)
        constraint.hidden = True
        return constraint
    if isinstance(constraint, str) and constraint == 'nan':
        return _NanConstraint()
    raise ValueError(f'Unknown constraint type: {constraint}')