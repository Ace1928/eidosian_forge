import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def signData(self, blob, data):
    """
        Request that the agent sign the given C{data} with the private key
        which corresponds to the public key given by C{blob}.  The private
        key should have been added to the agent already.

        @type blob: L{bytes}
        @type data: L{bytes}
        @return: A L{Deferred} which fires with a signature for given data
            created with the given key.
        """
    req = NS(blob)
    req += NS(data)
    req += b'\x00\x00\x00\x00'
    return self.sendRequest(AGENTC_SIGN_REQUEST, req).addCallback(self._cbSignData)