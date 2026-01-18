from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Circle, Polygon
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def makeSmiley(x, y, size, color):
    """Make a smiley marker."""
    d = size
    s = SmileyFace()
    s.fillColor = color
    s.x = x - d
    s.y = y - d
    s.size = d * 2
    return s