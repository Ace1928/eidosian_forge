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
def block_build(b, s):
    self.create_model(mod=b, model_option=ModelOptionLib.stage2)
    for par in self.param:
        cuid = pyo.ComponentUID(par)
        var = cuid.find_component_on(b)
        var.fix(self.scenario_data.scenario[s][par])