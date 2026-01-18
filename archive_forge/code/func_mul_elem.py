from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def mul_elem(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
    """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        When dealing with parametrized constant data, we need to repeat the variable tensor p times
        and stack them vertically to ensure shape compatibility for elementwise multiplication
        with the parametrized expression.
        """
    lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
    if is_param_free_lhs:

        def func(x, p):
            if p == 1:
                return lhs.multiply(x)
            else:
                new_lhs = sp.vstack([lhs] * p)
                return new_lhs.multiply(x)
    else:

        def parametrized_mul(x):
            return {k: v.multiply(sp.vstack([x] * self.param_to_size[k])) for k, v in lhs.items()}
        func = parametrized_mul
    return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)