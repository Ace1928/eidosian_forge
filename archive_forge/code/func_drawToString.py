from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawToString(d, msg='', showBoundary=rl_config._unset_, autoSize=1, canvasKwds={}):
    """Returns a PDF as a string in memory, without touching the disk"""
    s = BytesIO()
    drawToFile(d, s, msg=msg, showBoundary=showBoundary, autoSize=autoSize, canvasKwds=canvasKwds)
    return s.getvalue()