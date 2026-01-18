import sys
import pyomo.environ as pyo
import numpy.random as rnd
from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as po
def perform_estimation_external(data_fname, solver_trace=False):
    df = pd.read_csv(data_fname)
    npts = len(df)
    m = pyo.ConcreteModel()
    m.df = df
    m.PTS = pyo.Set(initialize=range(npts), ordered=True)

    def _model_i(b, i):
        po.build_single_point_model_external(b)
    m.model_i = pyo.Block(m.PTS, rule=_model_i)
    m.UA = pyo.Var()

    def _eq_parameter(m, i):
        return m.UA == m.model_i[i].egb.inputs['UA']
    m.eq_parameter = pyo.Constraint(m.PTS, rule=_eq_parameter)

    def _least_squares(m):
        obj = 0
        for i in m.PTS:
            row = m.df.iloc[i]
            obj += (m.model_i[i].egb.inputs['Th_in'] - float(row['Th_in'])) ** 2
            obj += (m.model_i[i].egb.inputs['Tc_in'] - float(row['Tc_in'])) ** 2
            obj += (m.model_i[i].egb.inputs['Th_out'] - float(row['Th_out'])) ** 2
            obj += (m.model_i[i].egb.inputs['Tc_out'] - float(row['Tc_out'])) ** 2
        return obj
    m.obj = pyo.Objective(rule=_least_squares)
    solver = pyo.SolverFactory('cyipopt')
    status, nlp = solver.solve(m, tee=solver_trace, return_nlp=True)
    if solver_trace:
        names = nlp.primals_names()
        values = nlp.evaluate_grad_objective()
        print({names[i]: values[i] for i in range(len(names))})
    return m