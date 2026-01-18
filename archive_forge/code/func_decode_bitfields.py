import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
def decode_bitfields(bits, r_mask, g_mask, b_mask, width, height, pitch, pitch_sign):
    r_shift1, r_shift2 = get_shift(r_mask)
    g_shift1, g_shift2 = get_shift(g_mask)
    b_shift1, b_shift2 = get_shift(b_mask)
    rgb_pitch = 3 * len(bits[0])
    buffer = (ctypes.c_ubyte * (height * rgb_pitch))()
    i = 0
    for row in bits:
        for packed in row:
            buffer[i] = (packed & r_mask) >> r_shift1 << r_shift2
            buffer[i + 1] = (packed & g_mask) >> g_shift1 << g_shift2
            buffer[i + 2] = (packed & b_mask) >> b_shift1 << b_shift2
            i += 3
    return ImageData(width, height, 'RGB', buffer, pitch_sign * rgb_pitch)