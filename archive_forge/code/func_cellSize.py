from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asBytes
from reportlab.platypus.paraparser import _num as paraparser_num
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isColor, isString, isColorOrNone, isNumber, isBoxAnchor
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import toColor
from reportlab.graphics.shapes import Group, Rect
@size.setter
def cellSize(self, v):
    self._cellSize = v
    self._recalc = True