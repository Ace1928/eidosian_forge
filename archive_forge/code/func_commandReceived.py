import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def commandReceived(self, command, argument):
    cmdFunc = self.commandMap.get(command)
    if cmdFunc is None:
        self.unhandledCommand(command, argument)
    else:
        cmdFunc(argument)