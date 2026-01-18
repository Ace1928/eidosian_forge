import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_SIGN_REQUEST(self, data):
    """
        Data is a structure with a reference to an already added key object and
        some data that the clients wants signed with that key.  If the key
        object wasn't loaded, return AGENT_FAILURE, else return the signature.
        """
    blob, data = getNS(data)
    if blob not in self.factory.keys:
        return self.sendResponse(AGENT_FAILURE, b'')
    signData, data = getNS(data)
    assert data == b'\x00\x00\x00\x00'
    self.sendResponse(AGENT_SIGN_RESPONSE, NS(self.factory.keys[blob][0].sign(signData)))