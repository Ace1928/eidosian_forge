from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import numpy as np
def make_invalid_op(name: str):
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str

    Returns
    -------
    invalid_op : function
    """

    def invalid_op(self, other=None):
        typ = type(self).__name__
        raise TypeError(f'cannot perform {name} with this index type: {typ}')
    invalid_op.__name__ = name
    return invalid_op