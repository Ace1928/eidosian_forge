from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def setupGLState(self):
    """
        This method is responsible for preparing the GL state options needed to render 
        this item (blending, depth testing, etc). The method is called immediately before painting the item.
        """
    for k, v in self.__glOpts.items():
        if v is None:
            continue
        if isinstance(k, str):
            func = getattr(GL, k)
            func(*v)
        elif v is True:
            glEnable(k)
        else:
            glDisable(k)