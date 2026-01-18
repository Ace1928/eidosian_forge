from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def readBool(text):
    if text.upper() in ('Y', 'YES', 'TRUE', '1'):
        return 1
    elif text.upper() in ('N', 'NO', 'FALSE', '0'):
        return 0
    else:
        raise ValueError("true/false attribute has illegal value '%s'" % text)