import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
def decode_24bit(bits, palette, width, height, pitch, pitch_sign):
    buffer = (ctypes.c_ubyte * (height * pitch))()
    ctypes.memmove(buffer, bits, len(buffer))
    return ImageData(width, height, 'BGR', buffer, pitch_sign * pitch)