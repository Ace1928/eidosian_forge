from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def mapFromParent(self, point):
    tr = self.transform()
    if tr is None:
        return point
    return tr.inverted()[0].map(point)