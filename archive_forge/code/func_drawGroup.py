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
def drawGroup(self, group):
    if self.verbose:
        print('### begin _SVGRenderer.drawGroup')
    currGroup = self._canvas.startGroup()
    a, b, c, d, e, f = self._tracker.getState()['transform']
    for childNode in group.getContents():
        if isinstance(childNode, UserNode):
            node2 = childNode.provideNode()
        else:
            node2 = childNode
        self.drawNode(node2)
    self._canvas.transform(a, b, c, d, e, f)
    self._canvas.endGroup(currGroup)
    if self.verbose:
        print('### end _SVGRenderer.drawGroup')