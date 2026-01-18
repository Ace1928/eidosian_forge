from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_badXPathNoClosingBracket(self) -> None:
    """
        A missing closing bracket raises a SyntaxError.

        This test excercises the most common failure mode.
        """
    exc = self.assertRaises(SyntaxError, XPathQuery, '//bar[@attrib1')
    self.assertTrue(exc.msg.startswith('Trying to find one of'), "SyntaxError message '%s' doesn't start with 'Trying to find one of'" % exc.msg)