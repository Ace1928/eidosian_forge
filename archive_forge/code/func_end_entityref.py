from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_entityref(self):
    self.gotEntityReference(self.erefbuf)