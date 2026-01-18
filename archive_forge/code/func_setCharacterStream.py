from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setCharacterStream(self, charfile):
    """Set the character stream for this input source. (The stream
        must be a Python 2.0 Unicode-wrapped file-like that performs
        conversion to Unicode strings.)

        If there is a character stream specified, the SAX parser will
        ignore any byte stream and will not attempt to open a URI
        connection to the system identifier."""
    self.__charfile = charfile