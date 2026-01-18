from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@property
def headers_end(self):
    """Calculates the byte offset for the end of the message headers.

        :rtype: int
        :returns: The byte offset from the beginning of the event stream
        message to the end of the headers.
        """
    return _PRELUDE_LENGTH + self.headers_length