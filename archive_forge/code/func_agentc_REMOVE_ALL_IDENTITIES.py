import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_REMOVE_ALL_IDENTITIES(self, data):
    """
        Remove all keys from the agent's collection of identities.
        """
    assert data == b''
    self.factory.keys = {}
    self.sendResponse(AGENT_SUCCESS, b'')