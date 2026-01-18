import unittest
import cvxpy as cp
from cvxpy.error import DCPError
from cvxpy.expressions.variable import Variable
def test_maximize(self) -> None:
    exp = self.x + self.z
    obj = cp.Maximize(exp)
    self.assertEqual(str(obj), 'maximize %s' % exp.name())
    new_obj, constraints = obj.canonical_form
    self.assertEqual(len(constraints), 0)
    with self.assertRaises(Exception) as cm:
        cp.Maximize(self.y).canonical_form
    self.assertEqual(str(cm.exception), "The 'maximize' objective must resolve to a scalar.")
    copy = obj.copy()
    self.assertTrue(type(copy) is type(obj))
    self.assertEqual(copy.args, obj.args)
    self.assertFalse(copy.args is obj.args)
    copy = obj.copy(args=[-cp.square(self.x)])
    self.assertTrue(type(copy) is type(obj))
    self.assertTrue(copy.args[0].args[0].args[0] is self.x)