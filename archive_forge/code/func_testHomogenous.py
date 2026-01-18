from twisted.python import roots
from twisted.trial import unittest
def testHomogenous(self) -> None:
    h = roots.Homogenous()
    h.entityType = int
    h.putEntity('a', 1)
    self.assertEqual(h.getStaticEntity('a'), 1)
    self.assertRaises(roots.ConstraintViolation, h.putEntity, 'x', 'y')