import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
def decode_1bit(bits, palette, width, height, pitch, pitch_sign):
    rgb_pitch = ((pitch << 3) + 7 & ~7) * 3
    buffer = (ctypes.c_ubyte * (height * rgb_pitch))()
    i = 0
    for row in bits:
        for packed in row:
            for _ in range(8):
                rgb = palette[(packed & 128) >> 7]
                buffer[i] = rgb.rgbRed
                buffer[i + 1] = rgb.rgbGreen
                buffer[i + 2] = rgb.rgbBlue
                i += 3
                packed <<= 1
    return ImageData(width, height, 'RGB', buffer, pitch_sign * rgb_pitch)