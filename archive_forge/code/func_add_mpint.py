import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_mpint(self, z):
    """
        Add a long int to the stream, encoded as an infinite-precision
        integer.  This method only works on positive numbers.

        :param int z: long int to add
        """
    self.add_string(util.deflate_long(z))
    return self