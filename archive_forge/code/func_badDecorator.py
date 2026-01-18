import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
def badDecorator(fn):
    """
    Decorate a function without preserving the name of the original function.
    Always return a function with the same name.
    """

    def nameCollision(*args, **kwargs):
        return fn(*args, **kwargs)
    return nameCollision