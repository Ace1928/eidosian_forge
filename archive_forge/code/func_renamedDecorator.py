import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
def renamedDecorator(self) -> None:
    """
        This is secretly a test method and will be decorated and then renamed so
        test discovery can find it.
        """