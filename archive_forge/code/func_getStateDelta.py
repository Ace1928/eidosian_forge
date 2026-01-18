from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def getStateDelta(shape):
    """Used to compute when we need to change the graphics state.
    For example, if we have two adjacent red shapes we don't need
    to set the pen color to red in between. Returns the effect
    the given shape would have on the graphics state"""
    delta = {}
    for prop, value in shape.getProperties().items():
        if prop in STATE_DEFAULTS:
            delta[prop] = value
    return delta