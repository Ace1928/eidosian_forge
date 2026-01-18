from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def jacobian_rule(m, p, n):
    """
            m: Pyomo model
            p: parameter
            n: response
            """
    cuid = pyo.ComponentUID(n)
    var_up = cuid.find_component_on(m.block[self.scenario_num[p][0]])
    var_lo = cuid.find_component_on(m.block[self.scenario_num[p][1]])
    if self.scale_nominal_param_value:
        return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.param[p] * self.scale_constant_value
    else:
        return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.scale_constant_value