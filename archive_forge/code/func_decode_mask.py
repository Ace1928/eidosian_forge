import json
import netaddr
import re
def decode_mask(mask_size):
    """Value/Mask decoder for values of specific size (bits).

    Used for fields such as:
        reg0=0x248/0xff
    """

    class Mask(IntMask):
        size = mask_size
        __name__ = 'Mask{}'.format(size)
    return Mask