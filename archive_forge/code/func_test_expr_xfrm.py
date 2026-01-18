import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_expr_xfrm(self):
    from pyomo.repn.plugins.gams_writer import expression_to_string, StorageTreeChecker
    from pyomo.core.expr.symbol_map import SymbolMap
    M = ConcreteModel()
    M.abc = Var()
    smap = SymbolMap()
    tc = StorageTreeChecker(M)
    expr = M.abc ** 2.0
    self.assertEqual(str(expr), 'abc**2.0')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('power(abc, 2)', False))
    expr = log(M.abc ** 2.0)
    self.assertEqual(str(expr), 'log(abc**2.0)')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('log(power(abc, 2))', False))
    expr = log(M.abc ** 2.0) + 5
    self.assertEqual(str(expr), 'log(abc**2.0) + 5')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('log(power(abc, 2)) + 5', False))
    expr = exp(M.abc ** 2.0) + 5
    self.assertEqual(str(expr), 'exp(abc**2.0) + 5')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('exp(power(abc, 2)) + 5', False))
    expr = log(M.abc ** 2.0) ** 4
    self.assertEqual(str(expr), 'log(abc**2.0)**4')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('power(log(power(abc, 2)), 4)', False))
    expr = log(M.abc ** 2.0) ** 4.5
    self.assertEqual(str(expr), 'log(abc**2.0)**4.5')
    self.assertEqual(expression_to_string(expr, tc, smap=smap), ('log(power(abc, 2)) ** 4.5', False))