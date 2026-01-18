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
def handleError(name, fmt):
    msg = 'Problem drawing %s fmt=%s file' % (name, fmt)
    if shout or verbose > 2:
        print(msg)
    errs.append('<br/><h2 style="color:red">%s</h2>' % msg)
    buf = StringIO()
    traceback.print_exc(file=buf)
    errs.append('<pre>%s</pre>' % escape(buf.getvalue()))