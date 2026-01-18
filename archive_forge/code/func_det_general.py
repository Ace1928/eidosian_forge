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
def det_general(m):
    """Calculate determinant. Can be applied to FIM of any size.
            det(A) = sum_{\\sigma \\in \\S_n} (sgn(\\sigma) * \\Prod_{i=1}^n a_{i,\\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
    r_list = list(range(len(m.regression_parameters)))
    object_p = permutations(r_list)
    list_p = list(object_p)
    det_perm = 0
    for i in range(len(list_p)):
        name_order = []
        x_order = list_p[i]
        for x in range(len(x_order)):
            for y, element in enumerate(m.regression_parameters):
                if x_order[x] == y:
                    name_order.append(element)
    det_perm = sum((self._sgn(list_p[d]) * sum((m.fim[each, name_order[b]] for b, each in enumerate(m.regression_parameters))) for d in range(len(list_p))))
    return m.det == det_perm