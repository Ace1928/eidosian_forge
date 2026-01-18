from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_waitscriptendtag(self, byte):
    self.temptagdata = ''
    self.tagName = ''
    self.endtag = 0