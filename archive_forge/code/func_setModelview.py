from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def setModelview(self):
    m = self.viewMatrix()
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(np.array(m.data(), dtype=np.float32))