import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_ADD_IDENTITY(self, data):
    """
        Adds a private key to the agent's collection of identities.  On
        subsequent interactions, the private key can be accessed using only the
        corresponding public key.
        """
    keyType, rest = getNS(data)
    if keyType == b'ssh-rsa':
        nmp = 6
    elif keyType == b'ssh-dss':
        nmp = 5
    else:
        raise keys.BadKeyError('unknown blob type: %s' % keyType)
    rest = getMP(rest, nmp)[-1]
    comment, rest = getNS(rest)
    k = keys.Key.fromString(data, type='private_blob')
    self.factory.keys[k.blob()] = (k, comment)
    self.sendResponse(AGENT_SUCCESS, b'')