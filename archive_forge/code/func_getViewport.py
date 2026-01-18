from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def getViewport(self):
    vp = self.opts['viewport']
    if vp is None:
        return (0, 0, self.deviceWidth(), self.deviceHeight())
    else:
        return vp