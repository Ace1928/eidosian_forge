import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def get_adaptive_int(self):
    """
        Fetch an int from the stream.

        :return: a 32-bit unsigned `int`.
        """
    byte = self.get_bytes(1)
    if byte == max_byte:
        return util.inflate_long(self.get_binary())
    byte += self.get_bytes(3)
    return struct.unpack('>I', byte)[0]