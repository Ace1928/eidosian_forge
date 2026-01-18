import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
def test_badDecorator(self) -> None:
    """
        This test method is decorated in a way that gives it a confusing name
        that collides with another method.
        """