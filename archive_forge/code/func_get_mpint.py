import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def get_mpint(self):
    """
        Fetch a long int (mpint) from the stream.

        :return: an arbitrary-length integer (`int`).
        """
    return util.inflate_long(self.get_binary())