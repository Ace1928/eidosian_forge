from OpenGL.GL import *  # noqa
from ... import functions as fn
from ...Qt import QtGui
from ..GLGraphicsItem import GLGraphicsItem
def setSize(self, x=None, y=None, z=None, size=None):
    """
        Set the size of the box (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
    if size is not None:
        x = size.x()
        y = size.y()
        z = size.z()
    self.__size = [x, y, z]
    self.update()