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
def fix1(mod, s):
    cuid = pyo.ComponentUID(name)
    design_var_global = cuid.find_component_on(mod)
    design_var = cuid.find_component_on(mod.block[s])
    return design_var == design_var_global