from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def shiftfont(self, program, face=None, bold=None, italic=None):
    oldface = self.face
    oldbold = self.bold
    olditalic = self.italic
    oldfontinfo = (oldface, oldbold, olditalic)
    if face is None:
        face = oldface
    if bold is None:
        bold = oldbold
    if italic is None:
        italic = olditalic
    self.face = face
    self.bold = bold
    self.italic = italic
    from reportlab.lib.fonts import tt2ps
    font = tt2ps(face, bold, italic)
    oldfont = tt2ps(oldface, oldbold, olditalic)
    if font != oldfont:
        program.append(('face', font))
    return oldfontinfo