import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def ssh_USERAUTH_PK_OK_keyboard_interactive(self, packet):
    """
        This is MSG_USERAUTH_INFO_RESPONSE.  The server has sent us the
        questions it wants us to answer, so we ask the user and sent the
        responses.
        """
    name, instruction, lang, data = getNS(packet, 3)
    numPrompts = struct.unpack('!L', data[:4])[0]
    data = data[4:]
    prompts = []
    for i in range(numPrompts):
        prompt, data = getNS(data)
        echo = bool(ord(data[0:1]))
        data = data[1:]
        prompts.append((prompt, echo))
    d = self.getGenericAnswers(name, instruction, prompts)
    d.addCallback(self._cbGenericAnswers)
    d.addErrback(self._ebAuth)