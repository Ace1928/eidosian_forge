from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def do_bullet(self, text, program):
    style = self.style1
    indent = style.bulletIndent + self.baseindent
    font = style.bulletFontName
    size = style.bulletFontSize
    program.append(('bullet', text, indent, font, size))