from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_attributeWithValueAny(self) -> None:
    """
        Test find nodes with attribute having value.
        """
    xp = XPathQuery("/foo/*[@attrib2='value2']")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar2])