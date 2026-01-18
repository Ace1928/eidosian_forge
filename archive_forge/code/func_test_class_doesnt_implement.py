import unittest
def test_class_doesnt_implement(self):
    from zope.interface import Interface
    from zope.interface.exceptions import DoesNotImplement

    class ICurrent(Interface):
        pass

    class Current:
        pass
    self.assertRaises(DoesNotImplement, self._callFUT, ICurrent, Current)