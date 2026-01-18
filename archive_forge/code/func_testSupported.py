from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
def testSupported(self):
    """
        Test supported version number 1.0
        """
    self.init.xmlstream.version = (1, 0)
    self.init.initialize()