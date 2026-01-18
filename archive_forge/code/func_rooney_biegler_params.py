from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def rooney_biegler_params(data):
    model = pyo.ConcreteModel()
    model.asymptote = pyo.Param(initialize=15, mutable=True)
    model.rate_constant = pyo.Param(initialize=0.5, mutable=True)

    def response_rule(m, h):
        expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
        return expr
    model.response_function = pyo.Expression(data.hour, rule=response_rule)
    return model