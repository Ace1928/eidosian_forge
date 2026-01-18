from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_unentity(self, byte):
    self.bodydata += byte