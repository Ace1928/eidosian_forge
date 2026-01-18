import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def dont_yes_false(self, state, option):
    state.us.state = 'no'
    self.disableLocal(option)
    self._wont(option)