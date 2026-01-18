import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_REQUEST_RSA_IDENTITIES(self, data):
    """
        v1 message for listing RSA1 keys; superseded by
        agentc_REQUEST_IDENTITIES, which handles different key types.
        """
    self.sendResponse(AGENT_RSA_IDENTITIES_ANSWER, struct.pack('!L', 0))