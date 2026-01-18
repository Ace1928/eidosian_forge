import unittest
def test_class_doesnt_have_required_method_simple(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.exceptions import BrokenImplementation

    class ICurrent(Interface):

        def method():
            """docstring"""

    @implementer(ICurrent)
    class Current:
        pass
    self.assertRaises(BrokenImplementation, self._callFUT, ICurrent, Current)