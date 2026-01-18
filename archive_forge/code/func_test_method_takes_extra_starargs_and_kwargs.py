import unittest
def test_method_takes_extra_starargs_and_kwargs(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):

        def method(a):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self, a, *args, **kw):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)