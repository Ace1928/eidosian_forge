import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def get_default_args(func):
    """Return optional arguments of a function.

    Parameters
    ----------
    func: callable

    Returns
    -------
    dict

    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}