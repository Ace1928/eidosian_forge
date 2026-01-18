import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
@property
def flow_controlled_length(self):
    """
        The length of the frame that needs to be accounted for when considering
        flow control.
        """
    padding_len = 0
    if 'PADDED' in self.flags:
        padding_len = self.total_padding + 1
    return len(self.data) + padding_len