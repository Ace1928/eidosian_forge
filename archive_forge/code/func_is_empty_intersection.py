import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
def is_empty_intersection(self, uncertain_params, nlp_solver):
    """
        Determine if intersection is empty.

        Arguments
        ---------
        uncertain_params : list of Param or list of Var
            List of uncertain parameter objects.
        nlp_solver : Pyomo SolverFactory object
            NLP solver.

        Returns
        -------
        is_empty_intersection : bool
            True if the intersection is certified to be empty,
            and False otherwise.
        """
    is_empty_intersection = True
    if any((a_set.type == 'discrete' for a_set in self.all_sets)):
        disc_sets = (a_set for a_set in self.all_sets if a_set.type == 'discrete')
        disc_set = min(disc_sets, key=lambda x: len(x.scenarios))
        for scenario in disc_set.scenarios:
            if all((a_set.point_in_set(point=scenario) for a_set in self.all_sets)):
                is_empty_intersection = False
                break
    else:
        m = ConcreteModel()
        m.obj = Objective(expr=0)
        m.param_vars = Var(uncertain_params.index_set())
        for a_set in self.all_sets:
            m.add_component(a_set.type + '_constraints', a_set.set_as_constraint(uncertain_params=m.param_vars))
        try:
            res = nlp_solver.solve(m)
        except:
            raise ValueError('Solver terminated with an error while checking set intersection non-emptiness.')
        if check_optimal_termination(res):
            is_empty_intersection = False
    return is_empty_intersection