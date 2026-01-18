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
def test_directlyProvides(self):

    class IA1(Interface):
        pass

    class IA2(Interface):
        pass

    class IB(Interface):
        pass

    class IC(Interface):
        pass

    class A(Odd):
        pass
    classImplements(A, IA1, IA2)

    class B(Odd):
        pass
    classImplements(B, IB)

    class C(A, B):
        pass
    classImplements(C, IC)
    ob = C()
    directlyProvides(ob, I1, I2)
    self.assertTrue(I1 in providedBy(ob))
    self.assertTrue(I2 in providedBy(ob))
    self.assertTrue(IA1 in providedBy(ob))
    self.assertTrue(IA2 in providedBy(ob))
    self.assertTrue(IB in providedBy(ob))
    self.assertTrue(IC in providedBy(ob))
    directlyProvides(ob, directlyProvidedBy(ob) - I2)
    self.assertTrue(I1 in providedBy(ob))
    self.assertFalse(I2 in providedBy(ob))
    self.assertFalse(I2 in providedBy(ob))
    directlyProvides(ob, directlyProvidedBy(ob), I2)
    self.assertTrue(I2 in providedBy(ob))