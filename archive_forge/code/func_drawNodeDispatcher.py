from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def drawNodeDispatcher(self, node):
    """dispatch on the node's (super) class: shared code"""
    canvas = getattr(self, '_canvas', None)
    try:
        node = _expandUserNode(node, canvas)
        if not node:
            return
        if hasattr(node, '_canvas'):
            ocanvas = 1
        else:
            node._canvas = canvas
            ocanvas = None
        self.fillDerivedValues(node)
        dtcb = getattr(node, '_drawTimeCallback', None)
        if dtcb:
            dtcb(node, canvas=canvas, renderer=self)
        if isinstance(node, Line):
            self.drawLine(node)
        elif isinstance(node, Image):
            self.drawImage(node)
        elif isinstance(node, Rect):
            self.drawRect(node)
        elif isinstance(node, Circle):
            self.drawCircle(node)
        elif isinstance(node, Ellipse):
            self.drawEllipse(node)
        elif isinstance(node, PolyLine):
            self.drawPolyLine(node)
        elif isinstance(node, Polygon):
            self.drawPolygon(node)
        elif isinstance(node, Path):
            self.drawPath(node)
        elif isinstance(node, String):
            self.drawString(node)
        elif isinstance(node, Group):
            self.drawGroup(node)
        elif isinstance(node, Wedge):
            self.drawWedge(node)
        elif isinstance(node, DirectDraw):
            node.drawDirectly(self)
        else:
            print('DrawingError', 'Unexpected element %s in drawing!' % str(node))
    finally:
        if not ocanvas:
            del node._canvas