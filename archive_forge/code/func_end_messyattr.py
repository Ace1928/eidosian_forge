from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_messyattr(self):
    if self.attrval:
        self.tagAttributes[self.attrname] = self.attrval