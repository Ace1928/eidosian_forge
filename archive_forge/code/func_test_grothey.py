import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import mpi4py_available, numpy_available
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
@unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
@unittest.skipIf(not numpy_available, 'numpy is not available.')
@unittest.skipIf(not ipopt_available, 'ipopt is not available.')
def test_grothey(self):

    def create_root():
        m = pyo.ConcreteModel()
        m.y = pyo.Var(bounds=(1, None))
        m.eta = pyo.Var(bounds=(-10, None))
        m.obj = pyo.Objective(expr=m.y ** 2 + m.eta)
        return m

    def create_subproblem(root):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=-m.x2)
        m.c1 = pyo.Constraint(expr=(m.x1 - 1) ** 2 + m.x2 ** 2 <= pyo.log(m.y))
        m.c2 = pyo.Constraint(expr=(m.x1 + 1) ** 2 + m.x2 ** 2 <= pyo.log(m.y))
        complicating_vars_map = pyo.ComponentMap()
        complicating_vars_map[root.y] = m.y
        return (m, complicating_vars_map)
    m = create_root()
    root_vars = [m.y]
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-08)
    m.benders.add_subproblem(subproblem_fn=create_subproblem, subproblem_fn_kwargs={'root': m}, root_eta=m.eta, subproblem_solver='ipopt')
    opt = pyo.SolverFactory('ipopt')
    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        if len(cuts_added) == 0:
            break
    self.assertAlmostEqual(m.y.value, 2.721381, 4)
    self.assertAlmostEqual(m.eta.value, -0.0337568, 4)