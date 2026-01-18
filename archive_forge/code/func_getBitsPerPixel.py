from ctypes import *
from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.darwin.cocoapy import CGDirectDisplayID, quartz, cf
from pyglet.libs.darwin.cocoapy import cfstring_to_string, cfarray_to_list
def getBitsPerPixel(self, cgmode):
    IO8BitIndexedPixels = 'PPPPPPPP'
    IO16BitDirectPixels = '-RRRRRGGGGGBBBBB'
    IO32BitDirectPixels = '--------RRRRRRRRGGGGGGGGBBBBBBBB'
    cfstring = c_void_p(quartz.CGDisplayModeCopyPixelEncoding(cgmode))
    pixelEncoding = cfstring_to_string(cfstring)
    cf.CFRelease(cfstring)
    if pixelEncoding == IO8BitIndexedPixels:
        return 8
    if pixelEncoding == IO16BitDirectPixels:
        return 16
    if pixelEncoding == IO32BitDirectPixels:
        return 32
    return 0