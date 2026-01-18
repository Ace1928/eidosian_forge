import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_add_incompatible(self) -> None:
    """Test addition of incompatible shapes raises a ValueError.
        """
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4, 2), (4,)])