from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def renderScaledDrawing(d):
    renderScale = d.renderScale
    if renderScale != 1.0:
        o = d
        d = d.__class__(o.width * renderScale, o.height * renderScale)
        d.__dict__ = o.__dict__.copy()
        d.scale(renderScale, renderScale)
        d.renderScale = 1.0
    return d