from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@property
def payload_length(self):
    """Calculates the total payload length.

        The extra minus 4 bytes is for the message CRC.

        :rtype: int
        :returns: The total payload length.
        """
    return self.total_length - self.headers_length - _PRELUDE_LENGTH - 4