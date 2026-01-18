from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
@staticmethod
def circleArcStream(size, r, arcs=(0, 1, 2, 3), rotated=False):
    R = [].append
    rlen = R.__self__.__len__
    hsize = size * 0.5
    f = size / 20.0
    size *= f
    hsize *= f
    r *= f
    cp = fp_str(0.55231 * r)
    r = fp_str(r)
    hsize = fp_str(hsize)
    mx = '0.7071 0.7071 -0.7071 0.7071' if rotated else '1 0 0 1'
    R('%(mx)s %(hsize)s %(hsize)s cm')
    if 0 in arcs:
        if rlen() == 1:
            R('%(r)s 0 m')
        R('%(r)s %(cp)s %(cp)s %(r)s 0 %(r)s c')
    if 1 in arcs:
        if rlen() == 1:
            R('0 %(r)s m')
        R('-%(cp)s %(r)s -%(r)s %(cp)s -%(r)s 0 c')
    if 2 in arcs:
        if rlen() == 1:
            R('-%(r)s 0 m')
        R('-%(r)s -%(cp)s -%(cp)s -%(r)s 0 -%(r)s c')
    if 3 in arcs:
        if rlen() == 1:
            R('0 -%(r)s m')
        R('%(cp)s -%(r)s %(r)s -%(cp)s %(r)s 0 c')
    return '\n'.join(R.__self__) % vars()