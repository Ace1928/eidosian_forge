import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_handle_empty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()

    @implementer(ifoo)
    class Bar:
        pass
    bar = Bar()
    comp.handle((bar,))