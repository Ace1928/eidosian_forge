from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def make_non_nested_model_declaring_Disjuncts_on_each_other():
    """
    T = {1, 2, ..., 10}

    min  sum(x_t + y_t for t in T)

    s.t. 1 <= x_t <= 10, for all t in T
         1 <= y_t <= 100, for all t in T

         [y_t = 100] v [y_t = 1000], for all t in T
         [x_t = 2] v [y_t = 10], for all t in T.


    We can't choose y_t = 10 because then the first Disjunction is infeasible.
    so in the optimal solution we choose x_t = 2 and y_t = 100 for all t in T.
    That gives us an optimal value of (100 + 2)*10 = 1020.
    """
    model = ConcreteModel()
    model.T = RangeSet(10)
    model.x = Var(model.T, bounds=(1, 10))
    model.y = Var(model.T, bounds=(1, 100))

    def _op_mode_sub(m, t):
        m.disj1[t].c1 = Constraint(expr=m.x[t] == 2)
        m.disj1[t].sub1 = Disjunct()
        m.disj1[t].sub1.c1 = Constraint(expr=m.y[t] == 100)
        m.disj1[t].sub2 = Disjunct()
        m.disj1[t].sub2.c1 = Constraint(expr=m.y[t] == 1000)
        return [m.disj1[t].sub1, m.disj1[t].sub2]

    def _op_mode(m, t):
        m.disj2[t].c1 = Constraint(expr=m.y[t] == 10)
        return [m.disj1[t], m.disj2[t]]
    model.disj1 = Disjunct(model.T)
    model.disj2 = Disjunct(model.T)
    model.disjunction1sub = Disjunction(model.T, rule=_op_mode_sub)
    model.disjunction1 = Disjunction(model.T, rule=_op_mode)

    def obj_rule(m, t):
        return sum((m.x[t] + m.y[t] for t in m.T))
    model.obj = Objective(rule=obj_rule)
    return model