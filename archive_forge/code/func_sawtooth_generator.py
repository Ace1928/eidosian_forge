import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def sawtooth_generator(frequency, sample_rate):
    period_length = int(sample_rate / frequency)
    step = 2 * frequency / sample_rate
    i = 0
    while True:
        yield (step * (i % period_length) - 1)
        i += 1