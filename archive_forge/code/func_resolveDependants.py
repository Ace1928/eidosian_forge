from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
def resolveDependants(self, newObject):
    self.resolved = 1
    self.resolvedObject = newObject
    for mut, key in self.dependants:
        mut[key] = newObject