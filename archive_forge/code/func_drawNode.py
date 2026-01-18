from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawNode(self, node):
    """This is the recursive method called for each node
        in the tree"""
    if not (isinstance(node, Path) and node.isClipPath):
        self._canvas.saveState()
    deltas = getStateDelta(node)
    self._tracker.push(deltas)
    self.applyStateChanges(deltas, {})
    self.drawNodeDispatcher(node)
    self._tracker.pop()
    if not (isinstance(node, Path) and node.isClipPath):
        self._canvas.restoreState()