from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def outerdisj_rule(d, flag):
    m = d.model()
    if flag:

        def innerdisj_rule(d, flag):
            m = d.model()
            if flag:
                d.c = Constraint(expr=m.x >= 2)
            else:
                d.c = Constraint(expr=m.x == 0)
        d.innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)
        d.innerdisjunction = Disjunction(expr=[d.innerdisjunct[0], d.innerdisjunct[1]])
        d.duplicateddisjunction = Disjunction(expr=[d.innerdisjunct[0], d.innerdisjunct[1]])
    else:
        d.c = Constraint(expr=m.x == 8)