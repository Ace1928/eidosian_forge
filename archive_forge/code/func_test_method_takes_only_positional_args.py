import unittest
def test_method_takes_only_positional_args(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):

        def method(a):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self, *args):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)