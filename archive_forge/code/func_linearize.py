from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def linearize(self, program=None, parsedText=None):
    if parsedText is None:
        parsedText = self.parsedText
    style = self.style1
    if program is None:
        program = []
    program.append(('push',))
    if style.spaceBefore:
        program.append(('leading', style.spaceBefore + style.leading))
    else:
        program.append(('leading', style.leading))
    program.append(('nextLine', 0))
    program = self.compileProgram(parsedText, program=program)
    program.append(('pop',))
    program.append(('push',))
    if style.spaceAfter:
        program.append(('leading', style.spaceAfter))
    else:
        program.append(('leading', 0))
    program.append(('nextLine', 0))
    program.append(('pop',))