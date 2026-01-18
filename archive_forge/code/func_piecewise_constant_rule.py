from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set
def piecewise_constant_rule(m, i, t):
    if t in sample_point_set:
        return Constraint.Skip
    else:
        var = inputs[i]
        if use_next:
            t_next = time.next(t)
            return var[t] - var[t_next] == 0
        else:
            t_prev = time.prev(t)
            return var[t_prev] - var[t] == 0