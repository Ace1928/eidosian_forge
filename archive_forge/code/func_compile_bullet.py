from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_bullet(self, attdict, content, extra, program):
    if len(content) != 1 or not isinstance(content[0], str):
        raise ValueError('content for bullet must be a single string')
    text = content[0]
    self.do_bullet(text, program)