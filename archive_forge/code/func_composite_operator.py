import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def composite_operator(*operators):
    return (sum(samples) / len(samples) for samples in zip(*operators))