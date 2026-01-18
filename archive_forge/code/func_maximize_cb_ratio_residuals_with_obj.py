import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_residuals import (
def maximize_cb_ratio_residuals_with_obj(show_solver_log=False, additional_options={}):
    m = pyo.ConcreteModel()
    m.reactor = ExternalGreyBoxBlock()
    m.reactor.set_external_model(ReactorModelNoOutputs())
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    u = m.reactor.inputs
    m.obj = pyo.Objective(expr=u['cb'] / (u['ca'] + u['cc'] + u['cd']), sense=pyo.maximize)
    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m