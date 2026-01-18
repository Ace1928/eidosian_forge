from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def shrinkWrap(self, line):
    result = []
    index = 0
    maxindex = len(line)
    while index < maxindex:
        e = line[index]
        if isinstance(e, str) and index < maxindex - 1:
            thestrings = [e]
            thefloats = 0.0
            index += 1
            nexte = line[index]
            while index < maxindex and isinstance(nexte, (float, str)):
                if isinstance(nexte, float):
                    if thefloats < 0 and nexte > 0:
                        thefloats = -thefloats
                    if nexte < 0 and thefloats > 0:
                        nexte = -nexte
                    thefloats += nexte
                elif isinstance(nexte, str):
                    thestrings.append(nexte)
                index += 1
                if index < maxindex:
                    nexte = line[index]
            s = ' '.join(thestrings)
            result.append(s)
            result.append(float(thefloats))
            index -= 1
        else:
            result.append(e)
        index += 1
    return result