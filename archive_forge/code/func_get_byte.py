import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def get_byte(self):
    """
        Return the next byte of the message, without decomposing it.  This
        is equivalent to `get_bytes(1) <get_bytes>`.

        :return:
            the next (`bytes`) byte of the message, or ``b'\x00'`` if there
            aren't any bytes remaining.
        """
    return self.get_bytes(1)