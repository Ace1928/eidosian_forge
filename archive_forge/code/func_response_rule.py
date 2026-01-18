from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
def response_rule(m, h):
    expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
    return expr