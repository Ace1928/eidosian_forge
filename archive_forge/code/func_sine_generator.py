import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def sine_generator(frequency, sample_rate):
    step = 2 * _math.pi * frequency / sample_rate
    i = 0
    while True:
        yield _math.sin(i * step)
        i += 1