from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def getStateValue(self, key):
    """Return current state parameter for given key"""
    currentState = self._tracker._combined[-1]
    return currentState[key]