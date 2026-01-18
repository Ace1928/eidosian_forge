import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def test_IfThen(self):
    m = ConcreteModel()
    m.w = Var(initialize=1)
    m.v = Var(initialize=0)
    m.v.fixed = True
    m.p = Param(mutable=True, initialize=1)
    e = EXPR.Expr_if(1, 1, m.w)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '1')
    e = EXPR.Expr_if(1, m.w, 0)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), 'w')
    e = EXPR.Expr_if(m.p == 0, 1, 0)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '0')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'Expr_if( ( p  ==  0 ), then=( 1 ), else=( 0 ) )')
    e = EXPR.Expr_if(m.p == 0, 1, m.v)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '0')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'Expr_if( ( p  ==  0 ), then=( 1 ), else=( v ) )')
    e = EXPR.Expr_if(m.v, 1, 0)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '0')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'Expr_if( ( v ), then=( 1 ), else=( 0 ) )')
    e = EXPR.Expr_if(m.w, 1, 0)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), 'Expr_if( ( w ), then=( 1 ), else=( 0 ) )')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'Expr_if( ( w ), then=( 1 ), else=( 0 ) )')