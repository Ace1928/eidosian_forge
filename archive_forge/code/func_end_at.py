from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def end_at(self, x, y, para, canvas, textobject):
    fontsize = para.fontSize
    rect = [self.xStart, self.yStart, x, y + fontsize]
    if debug:
        print('LINKING RECTANGLE', rect)
    self.link(rect, canvas)