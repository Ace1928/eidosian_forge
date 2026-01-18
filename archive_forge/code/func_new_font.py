import formatter
import string
from types import *
import htmllib
import piddle
def new_font(self, fontParams):
    if TRACE:
        print('nf', fontParams)
    if not fontParams:
        fontParams = (None, None, None, None)
    size = fontParams[0]
    try:
        points = self.FontSizeDict[size]
    except KeyError:
        points = self.DefaultFontSize
    if fontParams[3]:
        face = 'courier'
    elif isinstance(size, str) and size[0] == 'h':
        face = 'helvetica'
    else:
        face = 'times'
    italic = fontParams[1]
    if italic is None:
        italic = 0
    bold = fontParams[2]
    if bold is None:
        bold = 0
    self.font = piddle.Font(points, bold, italic, face=face)
    x = self.pc.stringWidth('W' * 20, self.font)
    self.fsizex = (x + 19) / 20
    self.fsizey = self.pc.fontHeight(self.font)