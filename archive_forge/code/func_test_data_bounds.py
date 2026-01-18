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
def test_data_bounds(self):
    A = numpy.ones((5, 4))
    ctuple = matrix_constraint(A)
    self.assertTrue((ctuple.lb == -numpy.inf).all())
    self.assertTrue((ctuple.ub == numpy.inf).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, None)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    ctuple.lb = 1
    ctuple.ub = 2
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 2).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 2)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    with self.assertRaises(ValueError):
        ctuple.lb = range(5)
    with self.assertRaises(ValueError):
        ctuple.lb = numpy.array(range(6))
    with self.assertRaises(ValueError):
        ctuple.ub = range(5)
    with self.assertRaises(ValueError):
        ctuple.ub = numpy.array(range(6))
    with self.assertRaises(ValueError):
        ctuple.equality = True
    for c in ctuple:
        with self.assertRaises(ValueError):
            ctuple.equality = True
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 2).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 2)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    for c in ctuple:
        c.bounds = (-1, 1)
    self.assertTrue((ctuple.lb == -1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, -1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.bounds, (-1, 1))
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    ctuple.lb = lb_ = -numpy.array(range(5))
    ctuple.ub = ub_ = numpy.array(range(5))
    self.assertTrue((ctuple.lb == lb_).all())
    self.assertTrue((ctuple.ub == ub_).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for i, c in enumerate(ctuple):
        self.assertEqual(c.lb, -i)
        self.assertEqual(c.ub, i)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    for c in ctuple:
        c.lb = None
        c.ub = None
    self.assertTrue((ctuple.lb == -numpy.inf).all())
    self.assertTrue((ctuple.ub == numpy.inf).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, None)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    for i, c in enumerate(ctuple):
        c.lb = -i
        c.ub = i
    self.assertTrue((ctuple.lb == lb_).all())
    self.assertTrue((ctuple.ub == ub_).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for i, c in enumerate(ctuple):
        self.assertEqual(c.lb, -i)
        self.assertEqual(c.ub, i)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    ctuple.lb = None
    ctuple.ub = None
    self.assertTrue((ctuple.lb == -numpy.inf).all())
    self.assertTrue((ctuple.ub == numpy.inf).all())
    self.assertTrue((ctuple.equality == False).all())
    with self.assertRaises(ValueError):
        ctuple.rhs
    for c in ctuple:
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, None)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.rhs
    ctuple.rhs = 1
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    for c in ctuple:
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)
    rhs_ = numpy.array(range(5))
    for i, c in enumerate(ctuple):
        c.rhs = i
    self.assertTrue((ctuple.lb == rhs_).all())
    self.assertTrue((ctuple.ub == rhs_).all())
    self.assertTrue((ctuple.ub == rhs_).all())
    self.assertTrue((ctuple.equality == True).all())
    for i, c in enumerate(ctuple):
        self.assertEqual(c.lb, i)
        self.assertEqual(c.ub, i)
        self.assertEqual(c.rhs, i)
        self.assertEqual(c.equality, True)
    with self.assertRaises(ValueError):
        ctuple.rhs = None
    with self.assertRaises(ValueError):
        ctuple.rhs = range(5)
    with self.assertRaises(ValueError):
        ctuple.rhs = numpy.array(range(6))
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = None
    self.assertTrue((ctuple.lb == rhs_).all())
    self.assertTrue((ctuple.ub == rhs_).all())
    self.assertTrue((ctuple.ub == rhs_).all())
    self.assertTrue((ctuple.equality == True).all())
    for i, c in enumerate(ctuple):
        self.assertEqual(c.lb, i)
        self.assertEqual(c.ub, i)
        self.assertEqual(c.rhs, i)
        self.assertEqual(c.equality, True)