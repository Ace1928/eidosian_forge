from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setByteStream(self, bytefile):
    """Set the byte stream (a Python file-like object which does
        not perform byte-to-character conversion) for this input
        source.

        The SAX parser will ignore this if there is also a character
        stream specified, but it will use a byte stream in preference
        to opening a URI connection itself.

        If the application knows the character encoding of the byte
        stream, it should set it with the setEncoding method."""
    self.__bytefile = bytefile