import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def eval_set(self, evals: Sequence[Tuple[DMatrix, str]], iteration: int=0, feval: Optional[Metric]=None, output_margin: bool=True) -> str:
    """Evaluate a set of data.

        Parameters
        ----------
        evals :
            List of items to be evaluated.
        iteration :
            Current iteration.
        feval :
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result string.
        """
    for d in evals:
        if not isinstance(d[0], DMatrix):
            raise TypeError(f'expected DMatrix, got {type(d[0]).__name__}')
        if not isinstance(d[1], str):
            raise TypeError(f'expected string, got {type(d[1]).__name__}')
        self._assign_dmatrix_features(d[0])
    dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
    evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
    msg = ctypes.c_char_p()
    _check_call(_LIB.XGBoosterEvalOneIter(self.handle, ctypes.c_int(iteration), dmats, evnames, c_bst_ulong(len(evals)), ctypes.byref(msg)))
    assert msg.value is not None
    res = msg.value.decode()
    if feval is not None:
        for dmat, evname in evals:
            feval_ret = feval(self.predict(dmat, training=False, output_margin=output_margin), dmat)
            if isinstance(feval_ret, list):
                for name, val in feval_ret:
                    res += '\t%s-%s:%f' % (evname, name, val)
            else:
                name, val = feval_ret
                res += '\t%s-%s:%f' % (evname, name, val)
    return res