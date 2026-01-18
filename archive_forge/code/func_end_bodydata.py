from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_bodydata(self):
    self.gotText(self.bodydata)
    self.bodydata = ''