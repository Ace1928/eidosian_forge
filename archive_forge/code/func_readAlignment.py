from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def readAlignment(text):
    up = text.upper()
    if up == 'LEFT':
        return TA_LEFT
    elif up == 'RIGHT':
        return TA_RIGHT
    elif up in ['CENTER', 'CENTRE']:
        return TA_CENTER
    elif up == 'JUSTIFY':
        return TA_JUSTIFY