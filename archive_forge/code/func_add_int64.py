import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_int64(self, n):
    """
        Add a 64-bit int to the stream.

        :param int n: long int to add
        """
    self.packet.write(struct.pack('>Q', n))
    return self