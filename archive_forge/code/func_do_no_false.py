import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def do_no_false(self, state, option):
    if self.enableLocal(option):
        state.us.state = 'yes'
        self._will(option)
    else:
        self._wont(option)