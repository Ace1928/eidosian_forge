import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def removeIdentity(self, blob):
    """
        Remove the private key corresponding to the public key in blob from the
        running agent.
        """
    req = NS(blob)
    return self.sendRequest(AGENTC_REMOVE_IDENTITY, req)