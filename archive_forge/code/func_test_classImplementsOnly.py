import unittest
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface import classImplementsOnly
from zope.interface import directlyProvidedBy
from zope.interface import directlyProvides
from zope.interface import implementedBy
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface.tests import odd
def test_classImplementsOnly(self):

    @implementer(I3)
    class A(Odd):
        pass

    @implementer(I4)
    class B(Odd):
        pass

    class C(A, B):
        pass
    classImplementsOnly(C, I1, I2)
    self.assertEqual([i.__name__ for i in implementedBy(C)], ['I1', 'I2'])