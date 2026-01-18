from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def saveToFile(self, fn, fmt=None):
    im = self.toPIL()
    if fmt is None:
        if not isinstance(fn, str):
            raise ValueError("Invalid value '%s' for fn when fmt is None" % ascii(fn))
        fmt = os.path.splitext(fn)[1]
        if fmt.startswith('.'):
            fmt = fmt[1:]
    configPIL = self.configPIL or {}
    configPIL.setdefault('preConvertCB', None)
    preConvertCB = configPIL.pop('preConvertCB')
    if preConvertCB:
        im = preConvertCB(im)
    fmt = fmt.upper()
    if fmt in ('GIF',):
        im = _convert2pilp(im)
    elif fmt in ('TIFF', 'TIFFP', 'TIFFL', 'TIF', 'TIFF1'):
        if fmt.endswith('P'):
            im = _convert2pilp(im)
        elif fmt.endswith('L'):
            im = _convert2pilL(im)
        elif fmt.endswith('1'):
            im = _convert2pil1(im)
        fmt = 'TIFF'
    elif fmt in ('PCT', 'PICT'):
        return _saveAsPICT(im, fn, fmt, transparent=configPIL.get('transparent', None))
    elif fmt in ('PNG', 'BMP', 'PPM'):
        pass
    elif fmt in ('JPG', 'JPEG'):
        fmt = 'JPEG'
    else:
        raise RenderPMError('Unknown image kind %s' % fmt)
    if fmt == 'TIFF':
        tc = configPIL.get('transparent', None)
        if tc:
            from PIL import ImageChops, Image
            T = 768 * [0]
            for o, c in zip((0, 256, 512), tc.bitmap_rgb()):
                T[o + c] = 255
            im = Image.merge('RGBA', im.split() + (ImageChops.invert(im.point(T).convert('L').point(255 * [0] + [255])),))
        for a, d in (('resolution', self._dpi), ('resolution unit', 'inch')):
            configPIL[a] = configPIL.get(a, d)
    configPIL.setdefault('chops_invert', 0)
    if configPIL.pop('chops_invert'):
        from PIL import ImageChops
        im = ImageChops.invert(im)
    configPIL.setdefault('preSaveCB', None)
    preSaveCB = configPIL.pop('preSaveCB')
    if preSaveCB:
        im = preSaveCB(im)
    im.save(fn, fmt, **configPIL)
    if not hasattr(fn, 'write') and os.name == 'mac':
        from reportlab.lib.utils import markfilename
        markfilename(fn, ext=fmt)