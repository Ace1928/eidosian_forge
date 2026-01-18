from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def cameraParams(self):
    valid_keys = {'center', 'rotation', 'distance', 'fov', 'elevation', 'azimuth'}
    return {k: self.opts[k] for k in valid_keys}