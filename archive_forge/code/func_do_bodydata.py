from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_bodydata(self, byte):
    if byte == '<':
        return 'tagstart'
    if byte == '&':
        return 'entityref'
    self.bodydata += byte