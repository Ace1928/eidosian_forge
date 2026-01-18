from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_anyLocationQueryForString(self) -> None:
    """
        L{XPathQuery.queryForString} should raise a L{NotImplementedError}
        for any location.
        """
    xp = XPathQuery('//bar')
    self.assertRaises(NotImplementedError, xp.queryForString, None)