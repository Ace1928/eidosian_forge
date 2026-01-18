import unittest
def test_class_doesnt_implement_but_classImplements_later(self):
    from zope.interface import Interface
    from zope.interface import classImplements

    class ICurrent(Interface):
        pass

    class Current:
        pass
    classImplements(Current, ICurrent)
    self._callFUT(ICurrent, Current)