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
def test_classImplements(self):

    @implementer(I3)
    class A(Odd):
        pass

    @implementer(I4)
    class B(Odd):
        pass

    class C(A, B):
        pass
    classImplements(C, I1, I2)
    self.assertEqual([i.getName() for i in implementedBy(C)], ['I1', 'I2', 'I3', 'I4'])
    classImplements(C, I5)
    self.assertEqual([i.getName() for i in implementedBy(C)], ['I1', 'I2', 'I5', 'I3', 'I4'])