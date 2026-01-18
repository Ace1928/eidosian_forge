from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def makeMarker(name, **kw):
    if Marker._attrMap['kind'].validate(name):
        m = Marker(**kw)
        m.kind = name
    elif name[-5:] == '_Flag' and Flag._attrMap['kind'].validate(name[:-5]):
        m = Flag(**kw)
        m.kind = name[:-5]
        m.size = 10
    else:
        raise ValueError('Invalid marker name %s' % name)
    return m