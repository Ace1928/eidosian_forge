from OpenGL.GL import *  # noqa
import numpy as np
from ... import functions as fn
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
def genTexture(x, y):
    r = np.hypot(x - (w - 1) / 2.0, y - (w - 1) / 2.0)
    return 255 * (w / 2 - fn.clip_array(r, w / 2 - 1, w / 2))