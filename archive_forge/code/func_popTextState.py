from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def popTextState(self):
    state = self.textStateStack[-1]
    self.textStateStack = self.textStateStack[:-1]
    state = state[:]
    for var in self.TEXT_STATE_VARIABLES:
        val = state[0]
        del state[0]
        setattr(self, var, val)