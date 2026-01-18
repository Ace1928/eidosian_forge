from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def cleanProgram(self, line):
    """collapse adjacent spacings"""
    result = []
    last = 0
    for e in line:
        if isinstance(e, float):
            if last < 0 and e > 0:
                last = -last
            if e < 0 and last > 0:
                e = -e
            last = float(last) + e
        else:
            if abs(last) > TOOSMALLSPACE:
                result.append(last)
            result.append(e)
            last = 0
    if last:
        result.append(last)
    change = 1
    rline = list(range(len(result) - 1))
    while change:
        change = 0
        for index in rline:
            nextindex = index + 1
            this = result[index]
            next = result[nextindex]
            doswap = 0
            if isinstance(this, str) or isinstance(next, str) or hasattr(this, 'width') or hasattr(next, 'width'):
                doswap = 0
            elif isinstance(this, tuple):
                thisindicator = this[0]
                if isinstance(next, tuple):
                    nextindicator = next[0]
                    doswap = 0
                    if nextindicator == 'endLineOperation' and thisindicator != 'endLineOperation' and (thisindicator != 'lineOperation'):
                        doswap = 1
                elif isinstance(next, float):
                    if thisindicator == 'lineOperation':
                        doswap = 1
            if doswap:
                result[index] = next
                result[nextindex] = this
                change = 1
    return result