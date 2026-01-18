import unittest
def test_multiple_invalid(self):
    from zope.interface import Interface
    from zope.interface import classImplements
    from zope.interface.exceptions import BrokenImplementation
    from zope.interface.exceptions import DoesNotImplement
    from zope.interface.exceptions import MultipleInvalid

    class ISeveralMethods(Interface):

        def meth1(arg1):
            """Method 1"""

        def meth2(arg1):
            """Method 2"""

    class SeveralMethods:
        pass
    with self.assertRaises(MultipleInvalid) as exc:
        self._callFUT(ISeveralMethods, SeveralMethods)
    ex = exc.exception
    self.assertEqual(3, len(ex.exceptions))
    self.assertIsInstance(ex.exceptions[0], DoesNotImplement)
    self.assertIsInstance(ex.exceptions[1], BrokenImplementation)
    self.assertIsInstance(ex.exceptions[2], BrokenImplementation)
    classImplements(SeveralMethods, ISeveralMethods)
    SeveralMethods.meth1 = lambda self, arg1: 'Hi'
    with self.assertRaises(BrokenImplementation):
        self._callFUT(ISeveralMethods, SeveralMethods)