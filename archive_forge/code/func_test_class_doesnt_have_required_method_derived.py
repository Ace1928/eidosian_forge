import unittest
def test_class_doesnt_have_required_method_derived(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.exceptions import BrokenImplementation

    class IBase(Interface):

        def method():
            """docstring"""

    class IDerived(IBase):
        pass

    @implementer(IDerived)
    class Current:
        pass
    self.assertRaises(BrokenImplementation, self._callFUT, IDerived, Current)