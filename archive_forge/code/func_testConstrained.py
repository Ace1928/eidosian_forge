from twisted.python import roots
from twisted.trial import unittest
def testConstrained(self) -> None:

    class const(roots.Constrained):

        def nameConstraint(self, name: str) -> bool:
            return name == 'x'
    c = const()
    self.assertIsNone(c.putEntity('x', 'test'))
    self.assertRaises(roots.ConstraintViolation, c.putEntity, 'y', 'test')