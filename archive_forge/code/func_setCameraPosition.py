from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None, rotation=None):
    if rotation is not None:
        if elevation is not None:
            raise ValueError('cannot set both rotation and elevation')
        if azimuth is not None:
            raise ValueError('cannot set both rotation and azimuth')
    if pos is not None:
        self.opts['center'] = pos
    if distance is not None:
        self.opts['distance'] = distance
    if self.opts['rotationMethod'] == 'quaternion':
        if elevation is not None or azimuth is not None:
            eu = self.opts['rotation'].toEulerAngles()
            if azimuth is not None:
                eu.setZ(-azimuth - 90)
            if elevation is not None:
                eu.setX(elevation - 90)
            self.opts['rotation'] = QtGui.QQuaternion.fromEulerAngles(eu)
        if rotation is not None:
            self.opts['rotation'] = rotation
    else:
        if elevation is not None:
            self.opts['elevation'] = elevation
        if azimuth is not None:
            self.opts['azimuth'] = azimuth
        if rotation is not None:
            eu = rotation.toEulerAngles()
            self.opts['elevation'] = eu.x() + 90
            self.opts['azimuth'] = -eu.z() - 90
    self.update()