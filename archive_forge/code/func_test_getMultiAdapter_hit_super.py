import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getMultiAdapter_hit_super(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    class IFoo(Interface):
        pass

    @implementer(IBase)
    class Base:
        pass

    @implementer(IDerived)
    class Derived(Base):
        pass

    class AdapterBase:

        def __init__(self, context1, context2):
            self.context1 = context1
            self.context2 = context2

    class AdapterDerived(AdapterBase):
        pass
    comp = self._makeOne()
    comp.registerAdapter(AdapterDerived, (IDerived, IDerived), IFoo)
    comp.registerAdapter(AdapterBase, (IBase, IDerived), IFoo)
    self._should_not_change(comp)
    derived = Derived()
    adapter = comp.getMultiAdapter((derived, derived), IFoo)
    self.assertIsInstance(adapter, AdapterDerived)
    self.assertIs(adapter.context1, derived)
    self.assertIs(adapter.context2, derived)
    supe = super(Derived, derived)
    adapter = comp.getMultiAdapter((supe, derived), IFoo)
    self.assertIsInstance(adapter, AdapterBase)
    self.assertNotIsInstance(adapter, AdapterDerived)
    self.assertIs(adapter.context1, derived)
    self.assertIs(adapter.context2, derived)