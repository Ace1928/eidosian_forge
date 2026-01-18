from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_waitforendscript(self, byte):
    if byte == '<':
        return 'waitscriptendtag'
    self.bodydata += byte