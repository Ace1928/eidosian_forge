import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_REMOVE_RSA_IDENTITY(self, data):
    """
        v1 message for removing RSA1 keys; superseded by
        agentc_REMOVE_IDENTITY, which handles different key types.
        """
    self.sendResponse(AGENT_SUCCESS, b'')