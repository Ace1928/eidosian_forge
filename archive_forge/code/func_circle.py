from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def circle(self, x_cen, y_cen, r):
    """adds a circle to the path"""
    x1 = x_cen - r
    y1 = y_cen - r
    width = height = 2 * r
    self.ellipse(x1, y1, width, height)