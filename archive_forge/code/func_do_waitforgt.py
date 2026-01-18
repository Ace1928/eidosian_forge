from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_waitforgt(self, byte):
    if byte == '>':
        if self.endtag or not self.beExtremelyLenient:
            return 'bodydata'
        return self.maybeBodyData()