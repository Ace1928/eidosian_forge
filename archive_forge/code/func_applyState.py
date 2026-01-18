from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def applyState(self):
    s = self._tracker.getState()
    self._canvas.ctm = s['ctm']
    self._canvas.strokeWidth = s['strokeWidth']
    alpha = s['strokeOpacity']
    if alpha is not None:
        self._canvas.strokeOpacity = alpha
    self._canvas.setStrokeColor(s['strokeColor'])
    self._canvas.lineCap = s['strokeLineCap']
    self._canvas.lineJoin = s['strokeLineJoin']
    self._canvas.fillMode = s['fillMode']
    da = s['strokeDashArray']
    if not da:
        da = None
    else:
        if not isinstance(da, (list, tuple)):
            da = (da,)
        if len(da) != 2 or not isinstance(da[1], (list, tuple)):
            da = (0, da)
    self._canvas.dashArray = da
    alpha = s['fillOpacity']
    if alpha is not None:
        self._canvas.fillOpacity = alpha
    self._canvas.setFillColor(s['fillColor'])
    self._canvas.setFont(s['fontName'], s['fontSize'])