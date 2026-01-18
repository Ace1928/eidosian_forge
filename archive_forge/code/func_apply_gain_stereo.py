import sys
import math
import array
from .utils import (
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration
@register_pydub_effect
def apply_gain_stereo(seg, left_gain=0.0, right_gain=0.0):
    """
    left_gain - amount of gain to apply to the left channel (in dB)
    right_gain - amount of gain to apply to the right channel (in dB)
    
    note: mono audio segments will be converted to stereo
    """
    if seg.channels == 1:
        left = right = seg
    elif seg.channels == 2:
        left, right = seg.split_to_mono()
    l_mult_factor = db_to_float(left_gain)
    r_mult_factor = db_to_float(right_gain)
    left_data = audioop.mul(left._data, left.sample_width, l_mult_factor)
    left_data = audioop.tostereo(left_data, left.sample_width, 1, 0)
    right_data = audioop.mul(right._data, right.sample_width, r_mult_factor)
    right_data = audioop.tostereo(right_data, right.sample_width, 0, 1)
    output = audioop.add(left_data, right_data, seg.sample_width)
    return seg._spawn(data=output, overrides={'channels': 2, 'frame_width': 2 * seg.sample_width})