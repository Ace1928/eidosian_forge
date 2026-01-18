from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNestedDisjunctions():
    """Three-term SimpleDisjunction built from two IndexedDisjuncts and one
    SimpleDisjunct. The SimpleDisjunct and one of the DisjunctDatas each
    contain a nested SimpleDisjunction (the disjuncts of which are declared
    on the same disjunct as the disjunction).

    (makeNestedDisjunctions_NestedDisjuncts is a much simpler model. All
    this adds is that it has a nested disjunction on a DisjunctData as well
    as on a ScalarDisjunct. So mostly it exists for historical reasons.)
    """
    m = ConcreteModel()
    m.x = Var(bounds=(-9, 9))
    m.z = Var(bounds=(0, 10))
    m.a = Var(bounds=(0, 23))

    def disjunct_rule(disjunct, flag):
        m = disjunct.model()
        if flag:

            def innerdisj_rule(disjunct, flag):
                m = disjunct.model()
                if flag:
                    disjunct.c = Constraint(expr=m.z >= 5)
                else:
                    disjunct.c = Constraint(expr=m.z == 0)
            disjunct.innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)

            @disjunct.Disjunction([0])
            def innerdisjunction(b, i):
                return [b.innerdisjunct[0], b.innerdisjunct[1]]
            disjunct.c = Constraint(expr=m.a <= 2)
        else:
            disjunct.c = Constraint(expr=m.x == 2)
    m.disjunct = Disjunct([0, 1], rule=disjunct_rule)

    def simpledisj_rule(disjunct):
        m = disjunct.model()

        @disjunct.Disjunct()
        def innerdisjunct0(disjunct):
            disjunct.c = Constraint(expr=m.x <= 2)

        @disjunct.Disjunct()
        def innerdisjunct1(disjunct):
            disjunct.c = Constraint(expr=m.x >= 4)
        disjunct.innerdisjunction = Disjunction(expr=[disjunct.innerdisjunct0, disjunct.innerdisjunct1])
    m.simpledisjunct = Disjunct(rule=simpledisj_rule)
    m.disjunction = Disjunction(expr=[m.simpledisjunct, m.disjunct[0], m.disjunct[1]])
    return m