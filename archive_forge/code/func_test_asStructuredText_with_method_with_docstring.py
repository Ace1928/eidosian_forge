import sys
import unittest
def test_asStructuredText_with_method_with_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod() -- This method is documented.', ''])

    class IHasMethod(Interface):
        """ This interface has a method.
            """

        def aMethod():
            """This method is documented.
                """
    self.assertEqual(self._callFUT(IHasMethod), EXPECTED)