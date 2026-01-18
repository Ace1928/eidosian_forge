import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.matrix_constraint import matrix_constraint, _MatrixConstraintData
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression
from pyomo.core.kernel.block import block, block_list
def test_canonical_form_sparse(self):
    A = numpy.array([[0, 2]])
    vlist = _create_variable_list(2)
    ctuple = matrix_constraint(A, x=vlist)
    self.assertEqual(ctuple.sparse, True)
    for c in ctuple:
        self.assertEqual(c._linear_canonical_form, True)
    terms = list(ctuple[0].terms)
    vs, cs = zip(*terms)
    self.assertEqual(len(terms), 1)
    self.assertIs(vs[0], vlist[1])
    self.assertEqual(cs[0], 2)
    repn = ctuple[0].canonical_form()
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], vlist[1])
    self.assertEqual(repn.linear_coefs, (2,))
    self.assertEqual(repn.constant, 0)
    vlist[0].fix(1)
    repn = ctuple[0].canonical_form()
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], vlist[1])
    self.assertEqual(repn.linear_coefs, (2,))
    self.assertEqual(repn.constant, 0)
    vlist[1].fix(2)
    repn = ctuple[0].canonical_form()
    self.assertEqual(repn.linear_vars, ())
    self.assertEqual(repn.linear_coefs, ())
    self.assertEqual(repn.constant, 4)
    repn = ctuple[0].canonical_form(compute_values=False)
    self.assertEqual(repn.linear_vars, ())
    self.assertEqual(repn.linear_coefs, ())
    self.assertEqual(repn.constant(), 4)