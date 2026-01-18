from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def arc(self, x1, y1, x2, y2, startAng=0, extent=90):
    """Contributed to piddlePDF by Robert Kern, 28/7/99.
        Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2,
        starting at startAng degrees and covering extent degrees.   Angles
        start with 0 to the right (+x) and increase counter-clockwise.
        These should have x1<x2 and y1<y2.

        The algorithm is an elliptical generalization of the formulae in
        Jim Fitzsimmon's TeX tutorial <URL: http://www.tinaja.com/bezarc1.pdf>."""
    self._curves(pdfgeom.bezierArc(x1, y1, x2, y2, startAng, extent))