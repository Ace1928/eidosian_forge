import unittest
def test_class_has_required_method_simple(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):

        def method():
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)