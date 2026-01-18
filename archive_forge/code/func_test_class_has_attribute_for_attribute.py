import unittest
def test_class_has_attribute_for_attribute(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):
        attr = Attribute('The foo Attribute')

    @implementer(ICurrent)
    class Current:
        attr = 1
    self._callFUT(ICurrent, Current)