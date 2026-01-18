from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_locationWithValueUnicode(self) -> None:
    """
        Nodes' attributes can be matched with non-ASCII values.
        """
    xp = XPathQuery("/foo/*[@attrib6='á']")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar7])