from typing import Callable, List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
def get_exprval_in_vec_func(ineq_form: bool) -> Callable:
    if ineq_form:
        return exprval_in_vec_ineq
    else:
        return exprval_in_vec_eq