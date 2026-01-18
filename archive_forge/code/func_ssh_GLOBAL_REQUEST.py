import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_GLOBAL_REQUEST(self, packet):
    """
        The other side has made a global request.  Payload::
            string  request type
            bool    want reply
            <request specific data>

        This dispatches to self.gotGlobalRequest.
        """
    requestType, rest = common.getNS(packet)
    wantReply, rest = (ord(rest[0:1]), rest[1:])
    ret = self.gotGlobalRequest(requestType, rest)
    if wantReply:
        reply = MSG_REQUEST_FAILURE
        data = b''
        if ret:
            reply = MSG_REQUEST_SUCCESS
            if isinstance(ret, (tuple, list)):
                data = ret[1]
        self.transport.sendPacket(reply, data)