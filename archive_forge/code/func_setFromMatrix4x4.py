from math import atan2, degrees
import numpy as np
from . import SRTTransform3D
from .Point import Point
from .Qt import QtGui
def setFromMatrix4x4(self, m):
    m = SRTTransform3D.SRTTransform3D(m)
    angle, axis = m.getRotation()
    if angle != 0 and (axis[0] != 0 or axis[1] != 0 or axis[2] != 1):
        print('angle: %s  axis: %s' % (str(angle), str(axis)))
        raise Exception('Can only convert 4x4 matrix to 3x3 if rotation is around Z-axis.')
    self._state = {'pos': Point(m.getTranslation()), 'scale': Point(m.getScale()), 'angle': angle}
    self.update()