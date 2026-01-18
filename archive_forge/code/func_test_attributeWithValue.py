from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_attributeWithValue(self) -> None:
    """
        Test matching node with attribute having value.
        """
    xp = XPathQuery("/foo[@attrib1='value1']")
    self.assertEqual(xp.matches(self.e), 1)