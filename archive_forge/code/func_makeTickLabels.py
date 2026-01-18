from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
def makeTickLabels(self):
    g = Group()
    if not self.visibleLabels:
        return g
    f = self._labelTextFormat
    if f is None:
        f = self.labelTextFormat or (self._allIntTicks() and '%.0f' or _defaultLabelFormatter)
    elif f is str and self._allIntTicks():
        f = '%.0f'
    elif hasattr(f, 'calcPlaces'):
        f.calcPlaces(self._tickValues)
    post = self.labelTextPostFormat
    scl = self.labelTextScale
    pos = [self._x, self._y]
    d = self._dataIndex
    pos[1 - d] = self._labelAxisPos()
    labels = self.labels
    if self.skipEndL != 'none':
        if self.isXAxis:
            sk = self._x
        else:
            sk = self._y
        if self.skipEndL == 'start':
            sk = [sk]
        else:
            sk = [sk, sk + self._length]
            if self.skipEndL == 'end':
                del sk[0]
    else:
        sk = []
    nticks = len(self._tickValues)
    nticks1 = nticks - 1
    for i, tick in enumerate(self._tickValues):
        label = i - nticks
        if label in labels:
            label = labels[label]
        else:
            label = labels[i]
        if f and label.visible:
            v = self.scale(tick)
            if sk:
                for skv in sk:
                    if abs(skv - v) < 1e-06:
                        v = None
                        break
            if v is not None:
                if scl is not None:
                    t = tick * scl
                else:
                    t = tick
                if isinstance(f, str):
                    txt = f % t
                elif isSeq(f):
                    if i < len(f):
                        txt = f[i]
                    else:
                        txt = ''
                elif hasattr(f, '__call__'):
                    if isinstance(f, TickLabeller):
                        txt = f(self, t)
                    else:
                        txt = f(t)
                else:
                    raise ValueError('Invalid labelTextFormat %s' % f)
                if post:
                    txt = post % txt
                pos[d] = v
                label.setOrigin(*pos)
                label.setText(txt)
                if self.keepTickLabelsInside:
                    if isinstance(self, XValueAxis):
                        a_x = self._x
                        if not i:
                            x0, y0, x1, y1 = label.getBounds()
                            if x0 < a_x:
                                label = label.clone(dx=label.dx + a_x - x0)
                        if i == nticks1:
                            a_x1 = a_x + self._length
                            x0, y0, x1, y1 = label.getBounds()
                            if x1 > a_x1:
                                label = label.clone(dx=label.dx - x1 + a_x1)
                g.add(label)
    return g