from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def naws(self, data):
    """
        Received NAWS. (Window dimensions.)
        """
    if len(data) == 4:
        columns, rows = struct.unpack(str('!HH'), data)
        self.size_received_callback(rows, columns)
    else:
        logger.warning('Wrong number of NAWS bytes')