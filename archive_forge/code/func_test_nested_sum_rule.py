import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_nested_sum_rule(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.J = RangeSet(3)
    m.K = Set(m.I, initialize={1: [10], 2: [10, 20], 3: [10, 20, 30]})
    m.x = Var(m.I, m.J, [10, 20, 30])

    @m.Constraint()
    def c(m):
        return sum((sum((m.x[i, j, k] for k in m.K[i])) for j in m.J for i in m.I)) <= 0
    template, indices = templatize_constraint(m.c)
    self.assertEqual(len(indices), 0)
    self.assertEqual(template.to_string(verbose=True), 'templatesum(templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), iter(_1, J), iter(_2, I))  <=  0')
    self.assertEqual(str(template), 'SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) for _1 in J for _2 in I)  <=  0')
    self.assertEqual(str(resolve_template(template)), 'x[1,1,10] + (x[2,1,10] + x[2,1,20]) + (x[3,1,10] + x[3,1,20] + x[3,1,30]) + (x[1,2,10]) + (x[2,2,10] + x[2,2,20]) + (x[3,2,10] + x[3,2,20] + x[3,2,30]) + (x[1,3,10]) + (x[2,3,10] + x[2,3,20]) + (x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0')