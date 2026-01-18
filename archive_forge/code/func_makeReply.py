import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
def makeReply(self, reply, version=0, port=0, ip='0.0.0.0'):
    self.transport.write(struct.pack('!BBH', version, reply, port) + socket.inet_aton(ip))
    if reply != 90:
        self.transport.loseConnection()