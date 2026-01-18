import pyomo.environ as pyo
import numpy.random as rnd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as pm
from pyomo.common.dependencies import pandas as pd
def generate_data_external(N, UA_mean, UA_std, seed=42):
    rnd.seed(seed)
    m = pyo.ConcreteModel()
    pm.build_single_point_model_external(m)
    m.UA_spec = pyo.Param(initialize=200, mutable=True)
    m.Th_in_spec = pyo.Param(initialize=100, mutable=True)
    m.Tc_in_spec = pyo.Param(initialize=30, mutable=True)
    m.UA_spec_con = pyo.Constraint(expr=m.egb.inputs['UA'] == m.UA_spec)
    m.Th_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Th_in'] == m.Th_in_spec)
    m.Tc_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Tc_in'] == m.Tc_in_spec)
    m.obj = pyo.Objective(expr=(m.egb.inputs['UA'] - m.UA_spec) ** 2)
    solver = pyo.SolverFactory('cyipopt')
    data = {'run': [], 'Th_in': [], 'Tc_in': [], 'Th_out': [], 'Tc_out': []}
    for i in range(N):
        UA = float(rnd.normal(UA_mean, UA_std))
        Th_in = 100 + float(rnd.normal(0, 2))
        Tc_in = 30 + float(rnd.normal(0, 2))
        m.UA_spec.value = UA
        m.Th_in_spec.value = Th_in
        m.Tc_in_spec.value = Tc_in
        status = solver.solve(m, tee=False)
        data['run'].append(i)
        data['Th_in'].append(pyo.value(m.egb.inputs['Th_in']))
        data['Tc_in'].append(pyo.value(m.egb.inputs['Tc_in']))
        data['Th_out'].append(pyo.value(m.egb.inputs['Th_out']))
        data['Tc_out'].append(pyo.value(m.egb.inputs['Tc_out']))
    return pd.DataFrame(data)