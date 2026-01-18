from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
def rooney_biegler_model_with_constraint(data):
    model = pyo.ConcreteModel()
    model.asymptote = pyo.Var(initialize=15)
    model.rate_constant = pyo.Var(initialize=0.5)
    model.response_function = pyo.Var(data.hour, initialize=0.0)

    def response_rule(m, h):
        return m.response_function[h] == m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
    model.response_function_constraint = pyo.Constraint(data.hour, rule=response_rule)

    def SSE_rule(m):
        return sum(((data.y[i] - m.response_function[data.hour[i]]) ** 2 for i in data.index))
    model.SSE = pyo.Objective(rule=SSE_rule, sense=pyo.minimize)
    return model