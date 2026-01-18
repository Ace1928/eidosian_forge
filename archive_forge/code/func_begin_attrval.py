from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_attrval(self, byte):
    self.quotetype = byte
    self.attrval = ''