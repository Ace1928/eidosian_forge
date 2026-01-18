from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_spacebodydata(self, byte):
    self.bodydata = self.erefextra
    self.erefextra = None