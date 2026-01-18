from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Circle, Polygon
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def makeEmptySquare(x, y, size, color):
    """Make an empty square marker."""
    d = size / 2.0
    rect = Rect(x - d, y - d, 2 * d, 2 * d)
    rect.strokeColor = color
    rect.fillColor = None
    return rect