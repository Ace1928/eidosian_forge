import unittest
import cvxpy as cp
from cvxpy.error import DCPError
from cvxpy.expressions.variable import Variable
def test_add_problems(self) -> None:
    """Test adding objectives.
        """
    expr1 = self.x ** 2
    expr2 = self.x ** (-1)
    alpha = 2
    assert (cp.Minimize(expr1) + cp.Minimize(expr2)).is_dcp()
    assert (cp.Maximize(-expr1) + cp.Maximize(-expr2)).is_dcp()
    with self.assertRaises(DCPError) as cm:
        cp.Minimize(expr1) + cp.Maximize(-expr2)
    self.assertEqual(str(cm.exception), 'Problem does not follow DCP rules.')
    assert (cp.Minimize(expr1) - cp.Maximize(-expr2)).is_dcp()
    assert (alpha * cp.Minimize(expr1)).is_dcp()
    assert (alpha * cp.Maximize(-expr1)).is_dcp()
    assert (-alpha * cp.Maximize(-expr1)).is_dcp()
    assert (-alpha * cp.Maximize(-expr1)).is_dcp()