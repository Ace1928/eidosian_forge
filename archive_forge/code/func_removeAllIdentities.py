import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def removeAllIdentities(self):
    """
        Remove all keys from the running agent.
        """
    return self.sendRequest(AGENTC_REMOVE_ALL_IDENTITIES, b'')