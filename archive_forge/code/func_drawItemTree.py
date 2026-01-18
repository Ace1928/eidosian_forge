from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def drawItemTree(self, item=None, useItemNames=False):
    if item is None:
        items = [x for x in self.items if x.parentItem() is None]
    else:
        items = item.childItems()
        items.append(item)
    items.sort(key=lambda a: a.depthValue())
    for i in items:
        if not i.visible():
            continue
        if i is item:
            try:
                glPushAttrib(GL_ALL_ATTRIB_BITS)
                if useItemNames:
                    glLoadName(i._id)
                    self._itemNames[i._id] = i
                i.paint()
            except:
                from .. import debug
                debug.printExc()
                print('Error while drawing item %s.' % str(item))
            finally:
                glPopAttrib()
        else:
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            try:
                tr = i.transform()
                glMultMatrixf(np.array(tr.data(), dtype=np.float32))
                self.drawItemTree(i, useItemNames=useItemNames)
            finally:
                glMatrixMode(GL_MODELVIEW)
                glPopMatrix()