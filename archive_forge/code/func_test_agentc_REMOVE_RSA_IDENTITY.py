import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_agentc_REMOVE_RSA_IDENTITY(self):
    """
        assert that we get the correct op code for an RSA remove identity request
        """
    d = self.client.sendRequest(agent.AGENTC_REMOVE_RSA_IDENTITY, b'')
    self.pump.flush()
    return d.addCallback(self.assertEqual, b'')