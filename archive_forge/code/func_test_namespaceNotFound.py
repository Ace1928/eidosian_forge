from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_namespaceNotFound(self) -> None:
    """
        Test not matching node with wrong namespace.
        """
    xp = XPathQuery("/foo[@xmlns='badns']/bar2")
    self.assertEqual(xp.matches(self.e), 0)