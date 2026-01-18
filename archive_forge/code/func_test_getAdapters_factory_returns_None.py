import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getAdapters_factory_returns_None(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    ibaz = IFoo('IBaz')

    @implementer(ibar)
    class _Context1:
        pass

    @implementer(ibaz)
    class _Context2:
        pass
    _context1 = _Context1()
    _context2 = _Context2()
    comp = self._makeOne()
    _called_with = []

    def _side_effect_only(context1, context2):
        _called_with.append((context1, context2))
        return None
    comp.registerAdapter(_side_effect_only, (ibar, ibaz), ifoo)
    self.assertEqual(list(comp.getAdapters((_context1, _context2), ifoo)), [])
    self.assertEqual(_called_with, [(_context1, _context2)])