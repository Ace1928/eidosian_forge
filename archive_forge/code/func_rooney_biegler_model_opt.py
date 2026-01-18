from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
def rooney_biegler_model_opt():
    """This function generates an instance of the rooney & biegler Pyomo model

    Returns
    -------
    m: an instance of the Pyomo model
        for uncertainty propagation
    """
    model = pyo.ConcreteModel()
    model.asymptote = pyo.Var(initialize=15)
    model.rate_constant = pyo.Var(initialize=0.5)
    model.obj = pyo.Objective(expr=model.asymptote * (1 - pyo.exp(-model.rate_constant * 10)), sense=pyo.minimize)
    return model