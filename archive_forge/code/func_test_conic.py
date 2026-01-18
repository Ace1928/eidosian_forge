import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_conic(self):
    model = pmo.block()
    model.o = pmo.objective(0.0)
    model.c = pmo.constraint(body=0.0, rhs=1)
    b = model.quadratic = pmo.block()
    b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
    b.r = pmo.variable(lb=0)
    b.c = pmo.conic.quadratic(x=b.x, r=b.r)
    model.o.expr += b.r
    model.c.body += b.r
    del b
    b = model.rotated_quadratic = pmo.block()
    b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
    b.r1 = pmo.variable(lb=0)
    b.r2 = pmo.variable(lb=0)
    b.c = pmo.conic.rotated_quadratic(x=b.x, r1=b.r1, r2=b.r2)
    model.o.expr += b.r1 + b.r2
    model.c.body += b.r1 + b.r2
    del b
    import mosek
    if mosek.Env().getversion() >= (9, 0, 0):
        b = model.primal_exponential = pmo.block()
        b.x1 = pmo.variable(lb=0)
        b.x2 = pmo.variable()
        b.r = pmo.variable(lb=0)
        b.c = pmo.conic.primal_exponential(x1=b.x1, x2=b.x2, r=b.r)
        model.o.expr += b.r
        model.c.body += b.r
        del b
        b = model.primal_power = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.r1 = pmo.variable(lb=0)
        b.r2 = pmo.variable(lb=0)
        b.c = pmo.conic.primal_power(x=b.x, r1=b.r1, r2=b.r2, alpha=0.6)
        model.o.expr += b.r1 + b.r2
        model.c.body += b.r1 + b.r2
        del b
        b = model.dual_exponential = pmo.block()
        b.x1 = pmo.variable()
        b.x2 = pmo.variable(ub=0)
        b.r = pmo.variable(lb=0)
        b.c = pmo.conic.dual_exponential(x1=b.x1, x2=b.x2, r=b.r)
        model.o.expr += b.r
        model.c.body += b.r
        del b
        b = model.dual_power = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.r1 = pmo.variable(lb=0)
        b.r2 = pmo.variable(lb=0)
        b.c = pmo.conic.dual_power(x=b.x, r1=b.r1, r2=b.r2, alpha=0.4)
        model.o.expr += b.r1 + b.r2
        model.c.body += b.r1 + b.r2
    if mosek.Env().getversion() >= (10, 0, 0):
        b = model.primal_geomean = pmo.block()
        b.r = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.x = pmo.variable()
        b.c = pmo.conic.primal_geomean(r=b.r, x=b.x)
        model.o.expr += b.r[0] + b.r[1]
        model.c.body += b.r[0] + b.r[1]
        del b
        b = model.dual_geomean = pmo.block()
        b.r = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.x = pmo.variable()
        b.c = pmo.conic.dual_geomean(r=b.r, x=b.x)
        model.o.expr += b.r[0] + b.r[1]
        model.c.body += b.r[0] + b.r[1]
        del b
        b = model.svec_psdcone = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(), pmo.variable(), pmo.variable()))
        b.c = pmo.conic.svec_psdcone(x=b.x)
        model.o.expr += b.x[0] + 2 * b.x[1] + b.x[2]
        model.c.body += b.x[0] + 2 * b.x[1] + b.x[2]
        del b
    opt = pmo.SolverFactory('mosek_direct')
    results = opt.solve(model)
    self.assertEqual(results.solution.status, SolutionStatus.optimal)