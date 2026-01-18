from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_doctype(self, byte):
    self.doctype = byte