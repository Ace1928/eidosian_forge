from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
import os
from copy import deepcopy
from itertools import product
def get_con_name_repr(separation_model, con, with_orig_name=True, with_obj_name=True):
    """
    Get string representation of performance constraint
    and any other modeling components to which it has
    been mapped.

    Parameters
    ----------
    separation_model : ConcreteModel
        Separation model.
    con : ScalarConstraint or ConstraintData
        Constraint for which to get the representation.
    with_orig_name : bool, optional
        If constraint was added during construction of the
        separation problem (i.e. if the constraint is a member of
        in `separation_model.util.new_constraints`),
        include the name of the original constraint from which
        `perf_con` was created.
    with_obj_name : bool, optional
        Include name of separation model objective to which
        constraint is mapped. Applicable only to performance
        constraints of the separation problem.

    Returns
    -------
    str
        Constraint name representation.
    """
    qual_strs = []
    if with_orig_name:
        orig_con = separation_model.util.map_new_constraint_list_to_original_con.get(con, con)
        if orig_con is not con:
            qual_strs.append(f'originally {orig_con.name!r}')
    if with_obj_name:
        objectives_map = separation_model.util.map_obj_to_constr
        separation_obj = objectives_map[con]
        qual_strs.append(f'mapped to objective {separation_obj.name!r}')
    final_qual_str = f' ({', '.join(qual_strs)})' if qual_strs else ''
    return f'{con.name!r}{final_qual_str}'