import logging
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def add_model_soc_constr(self, model: ScipModel, variables: List, rows: Iterator, A: dok_matrix, b: np.ndarray) -> Tuple:
    """Adds SOC constraint to the model using the data from mat and vec.

        Return tuple contains (QConstr, list of Constr, and list of variables).
        """
    from pyscipopt.scip import quicksum
    expr_list = {i: [] for i in rows}
    for (i, j), c in A.items():
        v = variables[j]
        try:
            expr_list[i].append((c, v))
        except Exception:
            pass
    soc_vars = []
    for i in rows:
        lb = 0 if len(soc_vars) == 0 else None
        var = model.addVar(obj=0, name='soc_t_%d' % i, vtype=VariableTypes.CONTINUOUS, lb=lb, ub=None)
        soc_vars.append(var)
    lin_expr_list = [b[i] - quicksum((coeff * var for coeff, var in expr_list[i])) for i in rows]
    new_lin_constrs = [model.addCons(soc_vars[i] == lin_expr_list[i]) for i, _ in enumerate(lin_expr_list)]
    t_term = soc_vars[0] * soc_vars[0]
    x_term = quicksum([var * var for var in soc_vars[1:]])
    constraint = model.addCons(x_term <= t_term)
    return (constraint, new_lin_constrs, soc_vars)