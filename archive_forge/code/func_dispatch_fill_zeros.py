from numpy in the following ways:
from __future__ import annotations
import operator
import numpy as np
from pandas.core import roperator
def dispatch_fill_zeros(op, left, right, result):
    """
    Call _fill_zeros with the appropriate fill value depending on the operation,
    with special logic for divmod and rdivmod.

    Parameters
    ----------
    op : function (operator.add, operator.div, ...)
    left : object (np.ndarray for non-reversed ops)
        We have excluded ExtensionArrays here
    right : object (np.ndarray for reversed ops)
        We have excluded ExtensionArrays here
    result : ndarray

    Returns
    -------
    result : np.ndarray

    Notes
    -----
    For divmod and rdivmod, the `result` parameter and returned `result`
    is a 2-tuple of ndarray objects.
    """
    if op is divmod:
        result = (mask_zero_div_zero(left, right, result[0]), _fill_zeros(result[1], left, right))
    elif op is roperator.rdivmod:
        result = (mask_zero_div_zero(right, left, result[0]), _fill_zeros(result[1], right, left))
    elif op is operator.floordiv:
        result = mask_zero_div_zero(left, right, result)
    elif op is roperator.rfloordiv:
        result = mask_zero_div_zero(right, left, result)
    elif op is operator.mod:
        result = _fill_zeros(result, left, right)
    elif op is roperator.rmod:
        result = _fill_zeros(result, right, left)
    return result