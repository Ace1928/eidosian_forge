from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import pyomo.environ as pyo
from pyomo.repn.plugins.lp_writer import LPWriter
def test_deterministic_unordered_sets(self):
    ref = '\\* Source Pyomo model name=unknown *\\\n\nmin \no:\n+1 x(a)\n+1 x(aaaaa)\n+1 x(ooo)\n+1 x(z)\n\ns.t.\n\nc_l_c(a)_:\n+1 x(a)\n>= 1\n\nc_l_c(aaaaa)_:\n+1 x(aaaaa)\n>= 5\n\nc_l_c(ooo)_:\n+1 x(ooo)\n>= 3\n\nc_l_c(z)_:\n+1 x(z)\n>= 1\n\nbounds\n   -inf <= x(a) <= +inf\n   -inf <= x(aaaaa) <= +inf\n   -inf <= x(ooo) <= +inf\n   -inf <= x(z) <= +inf\nend\n'
    set_init = ['a', 'z', 'ooo', 'aaaaa']
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=set_init, ordered=False)
    m.x = pyo.Var(m.I)
    m.c = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] >= len(i))
    m.o = pyo.Objective(expr=sum((m.x[i] for i in m.I)))
    OUT = StringIO()
    with LoggingIntercept() as LOG:
        LPWriter().write(m, OUT, symbolic_solver_labels=True)
    self.assertEqual(LOG.getvalue(), '')
    print(OUT.getvalue())
    self.assertEqual(ref, OUT.getvalue())
    m = pyo.ConcreteModel()
    m.I = pyo.Set()
    m.x = pyo.Var(pyo.Any)
    m.c = pyo.Constraint(pyo.Any)
    for i in set_init:
        m.c[i] = m.x[i] >= len(i)
    m.o = pyo.Objective(expr=sum(m.x.values()))
    OUT = StringIO()
    with LoggingIntercept() as LOG:
        LPWriter().write(m, OUT, symbolic_solver_labels=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(ref, OUT.getvalue())