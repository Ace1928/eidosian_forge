from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def prefixedMethodClassDict(clazz, prefix):
    return {name: getattr(clazz, prefix + name) for name in prefixedMethodNames(clazz, prefix)}