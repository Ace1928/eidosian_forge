from numbers import Integral, Real
from .specs import (
def check_frame_value(value):
    if not isinstance(value, Integral):
        raise TypeError('frame_value must be int')
    elif not 0 <= value <= 15:
        raise ValueError('frame_value must be in range 0..15')