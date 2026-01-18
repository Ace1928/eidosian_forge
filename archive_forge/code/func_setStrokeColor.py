from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setStrokeColor(self, aColor, alpha=None):
    """Takes a color object, allowing colors to be referred to by name"""
    if self._enforceColorSpace:
        aColor = self._enforceColorSpace(aColor)
    if isinstance(aColor, CMYKColor):
        d = aColor.density
        c, m, y, k = (d * aColor.cyan, d * aColor.magenta, d * aColor.yellow, d * aColor.black)
        self._strokeColorObj = aColor
        name = self._checkSeparation(aColor)
        if name:
            self._code.append('/%s CS %s SCN' % (name, fp_str(d)))
        else:
            self._code.append('%s K' % fp_str(c, m, y, k))
    elif isinstance(aColor, Color):
        rgb = (aColor.red, aColor.green, aColor.blue)
        self._strokeColorObj = aColor
        self._code.append('%s RG' % fp_str(rgb))
    elif isinstance(aColor, (tuple, list)):
        l = len(aColor)
        if l == 3:
            self._strokeColorObj = aColor
            self._code.append('%s RG' % fp_str(aColor))
        elif l == 4:
            self._strokeColorObj = aColor
            self._code.append('%s K' % fp_str(aColor))
        else:
            raise ValueError('Unknown color %r' % aColor)
    elif isStr(aColor):
        self.setStrokeColor(toColor(aColor))
    else:
        raise ValueError('Unknown color %r' % aColor)
    if alpha is not None:
        self.setStrokeAlpha(alpha)
    elif getattr(aColor, 'alpha', None) is not None:
        self.setStrokeAlpha(aColor.alpha)