from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def moveTo(self, x, y):
    self._code_append('%s m' % fp_str(x, y))