from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def makeOuterRect(self):
    strokeColor = getattr(self, 'rectStrokeColor', self.strokeColor)
    strokeWidth = getattr(self, 'rectStrokeWidth', self.strokeWidth)
    if self.fillColor or (strokeColor and strokeWidth):
        rect = Rect(self.x, self.y, self.width, self.height)
        rect.fillColor = self.fillColor
        rect.strokeColor = strokeColor
        rect.strokeWidth = strokeWidth
        return rect
    else:
        return None