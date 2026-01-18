from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
def test_nodeprepUnassignedInUnicode32(self) -> None:
    """
        Make sure unassigned code points from Unicode 3.2 are rejected.
        """
    self.assertRaises(UnicodeError, nodeprep.prepare, 'ᴹ')