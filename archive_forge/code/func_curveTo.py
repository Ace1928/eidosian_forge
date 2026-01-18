from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def curveTo(self, x1, y1, x2, y2, x3, y3):
    self._code_append('%s c' % fp_str(x1, y1, x2, y2, x3, y3))