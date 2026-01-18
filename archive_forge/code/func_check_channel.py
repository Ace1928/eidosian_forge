from numbers import Integral, Real
from .specs import (
def check_channel(channel):
    if not isinstance(channel, Integral):
        raise TypeError('channel must be int')
    elif not 0 <= channel <= 15:
        raise ValueError('channel must be in range 0..15')