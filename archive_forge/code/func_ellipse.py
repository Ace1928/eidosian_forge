from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def ellipse(self, x, y, width, height):
    """adds an ellipse to the path"""
    self._curves(pdfgeom.bezierArc(x, y, x + width, y + height, 0, 360))