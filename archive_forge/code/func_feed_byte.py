from collections import deque
from numbers import Integral
from .messages.specs import SPEC_BY_STATUS, SYSEX_END, SYSEX_START
def feed_byte(self, byte):
    """Feed MIDI byte to the decoder.

        Takes an int in range [0..255].
        """
    if not isinstance(byte, Integral):
        raise TypeError('message byte must be integer')
    if 0 <= byte <= 255:
        if byte <= 127:
            return self._feed_data_byte(byte)
        else:
            return self._feed_status_byte(byte)
    else:
        raise ValueError(f'invalid byte value {byte!r}')