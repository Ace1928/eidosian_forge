from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_attrname(self, byte):
    self.attrname = byte
    self._attrname_termtag = 0