from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def applyStateChanges(self, delta, newState):
    """This takes a set of states, and outputs the PDF operators
        needed to set those properties"""
    for key, value in sorted(delta.items()) if rl_config.invariant else delta.items():
        if key == 'transform':
            self._canvas.transform(value[0], value[1], value[2], value[3], value[4], value[5])
        elif key == 'strokeColor':
            if value is None:
                self._stroke = 0
            else:
                self._stroke = 1
                self.setStrokeColor(value)
        elif key == 'strokeWidth':
            self._canvas.setLineWidth(value)
        elif key == 'strokeLineCap':
            self._canvas.setLineCap(value)
        elif key == 'strokeLineJoin':
            self._canvas.setLineJoin(value)
        elif key == 'strokeDashArray':
            if value:
                if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (tuple, list)):
                    phase = value[0]
                    value = value[1]
                else:
                    phase = 0
                self._canvas.setDash(value, phase)
            else:
                self._canvas.setDash()
        elif key == 'fillColor':
            if value is None:
                self._fill = 0
            else:
                self._fill = 1
                self.setFillColor(value)
        elif key in ['fontSize', 'fontName']:
            fontname = delta.get('fontName', self._canvas._fontname)
            fontsize = delta.get('fontSize', self._canvas._fontsize)
            self._canvas.setFont(fontname, fontsize)
        elif key == 'fillOpacity':
            if value is not None:
                self._canvas.setFillAlpha(value)
        elif key == 'strokeOpacity':
            if value is not None:
                self._canvas.setStrokeAlpha(value)
        elif key == 'fillOverprint':
            self._canvas.setFillOverprint(value)
        elif key == 'strokeOverprint':
            self._canvas.setStrokeOverprint(value)
        elif key == 'overprintMask':
            self._canvas.setOverprintMask(value)
        elif key == 'fillMode':
            self._canvas._fillMode = value