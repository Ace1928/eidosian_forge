from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_attrval(self):
    self.tagAttributes[self.attrname] = self.attrval
    self.attrname = self.attrval = ''