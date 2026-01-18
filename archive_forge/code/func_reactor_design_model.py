from pyomo.common.dependencies import pandas as pd
from pyomo.environ import (
def reactor_design_model(data):
    model = ConcreteModel()
    model.k1 = Param(initialize=5.0 / 6.0, within=PositiveReals, mutable=True)
    model.k2 = Param(initialize=5.0 / 3.0, within=PositiveReals, mutable=True)
    model.k3 = Param(initialize=1.0 / 6000.0, within=PositiveReals, mutable=True)
    if isinstance(data, dict) or isinstance(data, pd.Series):
        model.caf = Param(initialize=float(data['caf']), within=PositiveReals)
    elif isinstance(data, pd.DataFrame):
        model.caf = Param(initialize=float(data.iloc[0]['caf']), within=PositiveReals)
    else:
        raise ValueError('Unrecognized data type.')
    if isinstance(data, dict) or isinstance(data, pd.Series):
        model.sv = Param(initialize=float(data['sv']), within=PositiveReals)
    elif isinstance(data, pd.DataFrame):
        model.sv = Param(initialize=float(data.iloc[0]['sv']), within=PositiveReals)
    else:
        raise ValueError('Unrecognized data type.')
    model.ca = Var(initialize=5000.0, within=PositiveReals)
    model.cb = Var(initialize=2000.0, within=PositiveReals)
    model.cc = Var(initialize=2000.0, within=PositiveReals)
    model.cd = Var(initialize=1000.0, within=PositiveReals)
    model.obj = Objective(expr=model.cb, sense=maximize)
    model.ca_bal = Constraint(expr=0 == model.sv * model.caf - model.sv * model.ca - model.k1 * model.ca - 2.0 * model.k3 * model.ca ** 2.0)
    model.cb_bal = Constraint(expr=0 == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb)
    model.cc_bal = Constraint(expr=0 == -model.sv * model.cc + model.k2 * model.cb)
    model.cd_bal = Constraint(expr=0 == -model.sv * model.cd + model.k3 * model.ca ** 2.0)
    return model