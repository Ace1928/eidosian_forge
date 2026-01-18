import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def match_prediction(pixels, fmt, fd, pitch):
    assert len(fd['alpha']) == 2
    assert len(fd['pattern']) > 0
    bpp = bytes_per_pixel(fmt)
    rowlen = fd['w'] * bpp
    if pitch is None:
        pitch = rowlen + 3 & ~3
    elif pitch == 0:
        pitch = fd['w'] * bpp
    pitchalign = pitch - rowlen
    errors = []
    fail = errors.append
    if len(pixels) != pitch * fd['h']:
        fail('Pitch error: pitch {} * {} height != {} pixelbytes'.format(pitch, fd['h'], len(pixels)))
    ptr = 0
    pixnum = 0
    for char in fd['pattern']:
        pix = list(bytearray(pixels[ptr:ptr + bpp]))
        if len(pix) != bpp:
            fail('Want {} bytes per pixel, got {}: {}'.format(bpp, len(pix), pix))
            break
        if char == 't':
            if get_pixel_alpha(pix, fmt) != 0:
                fail("pixel {} nonzero 't' pixel alpha {:02X}: {}".format(pixnum, get_pixel_alpha(pix, fmt), pix))
        else:
            srcpix = v0_PIXELS[char] + list(bytearray.fromhex(fd['alpha']))
            predict = rgba_to(srcpix, fmt, 1, 1, pitch=0)
            predict = list(bytearray(predict))
            if not predict or not pix or predict != pix:
                fail('pixel {} {} format mismatch: want {} ({}) -- got {}'.format(pixnum, fmt, predict, char, pix))
        if pitchalign and (pixnum + 1) % fd['w'] == 0:
            check = list(bytearray(pixels[ptr + bpp:ptr + bpp + pitchalign]))
            if check != [0] * pitchalign:
                fail('Want {} 0x00 pitch align pixnum={}, pos={} got: {}'.format(pitchalign, pixnum, ptr + bpp, check))
            ptr += pitchalign
        ptr += bpp
        pixnum += 1
    if ptr != len(pixels):
        fail('Excess data: pixnum={} ptr={} bytes={}, bpp={} pitchalign={}'.format(pixnum, ptr, len(pixels), bpp, pitchalign))
    return (len(errors) == 0, errors)