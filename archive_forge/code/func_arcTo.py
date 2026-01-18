from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def arcTo(self, x1, y1, x2, y2, startAng=0, extent=90):
    """Like arc, but draws a line from the current point to
        the start if the start is not the current point."""
    self._curves(pdfgeom.bezierArc(x1, y1, x2, y2, startAng, extent), 'lineTo')