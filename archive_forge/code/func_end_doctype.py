from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_doctype(self):
    self.gotDoctype(self.doctype)
    self.doctype = None