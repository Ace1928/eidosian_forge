from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_super(self, attdict, content, extra, program):
    size = self.size
    self.size = newsize = size * 0.7
    rise = size * 0.5
    program.append(('size', newsize))
    program.append(('rise', rise))
    for e in content:
        self.compileComponent(e, program)
    program.append(('size', size))
    self.size = size
    program.append(('rise', -rise))