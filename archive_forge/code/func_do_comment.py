from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_comment(self, byte):
    self.commentbuf += byte
    if self.commentbuf.endswith('-->'):
        self.gotComment(self.commentbuf[:-3])
        return 'bodydata'