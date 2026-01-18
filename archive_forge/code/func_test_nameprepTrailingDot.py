from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
def test_nameprepTrailingDot(self) -> None:
    """
        A trailing dot in domain names is preserved.
        """
    self.assertEqual(nameprep.prepare('example.com.'), 'example.com.')