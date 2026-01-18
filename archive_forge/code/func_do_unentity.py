from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_unentity(self, byte):
    self.bodydata += byte
    return 'bodydata'