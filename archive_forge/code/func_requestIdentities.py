import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def requestIdentities(self):
    """
        @return: A L{Deferred} which will fire with a list of all keys found in
            the SSH agent. The list of keys is comprised of (public key blob,
            comment) tuples.
        """
    d = self.sendRequest(AGENTC_REQUEST_IDENTITIES, b'')
    d.addCallback(self._cbRequestIdentities)
    return d