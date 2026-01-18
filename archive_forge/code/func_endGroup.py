import math, sys, os, codecs, base64
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import stringWidth # for font info
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative
from reportlab.graphics.renderbase import getStateDelta, Renderer, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS, Path, UserNode
from reportlab.graphics.shapes import * # (only for test0)
from reportlab import rl_config
from reportlab.lib.utils import RLString, isUnicode, isBytes
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from .renderPM import _getImage
from xml.dom import getDOMImplementation
def endGroup(self, currGroup):
    if self.verbose:
        print('+++ begin SVGCanvas.endGroup')
    self.currGroup = currGroup
    if self.verbose:
        print('+++ end SVGCanvas.endGroup')