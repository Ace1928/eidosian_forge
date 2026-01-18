from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def lengthSequence(s, converter=readLength):
    """from "(2, 1)" or "2,1" return [2,1], for example"""
    s = s.strip()
    if s[:1] == '(' and s[-1:] == ')':
        s = s[1:-1]
    sl = s.split(',')
    sl = [s.strip() for s in sl]
    sl = [converter(s) for s in sl]
    return sl