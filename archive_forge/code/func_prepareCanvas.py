from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
def prepareCanvas(self, canvas):
    """You can ask a LineStyle to set up the canvas for drawing
        the lines."""
    canvas.setLineWidth(1)