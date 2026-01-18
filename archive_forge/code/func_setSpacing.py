from OpenGL.GL import *  # noqa
import numpy as np
from ... import QtGui
from ... import functions as fn
from ..GLGraphicsItem import GLGraphicsItem
def setSpacing(self, x=None, y=None, z=None, spacing=None):
    """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
    if spacing is not None:
        x = spacing.x()
        y = spacing.y()
        z = spacing.z()
    self.__spacing = [x, y, z]
    self.update()