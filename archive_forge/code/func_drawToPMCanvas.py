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
def drawToPMCanvas(d, dpi=72, bg=16777215, configPIL=None, showBoundary=rl_config._unset_, backend=rl_config.renderPMBackend):
    d = renderScaledDrawing(d)
    c = PMCanvas(d.width, d.height, dpi=dpi, bg=bg, configPIL=configPIL, backend=backend)
    draw(d, c, 0, 0, showBoundary=showBoundary)
    return c