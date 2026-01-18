import sys
import unittest
def test_asStructuredText_empty_with_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['IEmpty', ' This is an empty interface.', ' Attributes:', ' Methods:', ''])

    class IEmpty(Interface):
        """ This is an empty interface.
            """
    self.assertEqual(self._callFUT(IEmpty), EXPECTED)