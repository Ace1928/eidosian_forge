from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def prefixedMethodObjDict(obj, prefix):
    return {name: getattr(obj, prefix + name) for name in prefixedMethodNames(obj.__class__, prefix)}