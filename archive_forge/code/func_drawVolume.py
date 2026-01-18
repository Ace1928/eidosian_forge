from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from ..GLGraphicsItem import GLGraphicsItem
def drawVolume(self, ax, d):
    imax = [0, 1, 2]
    imax.remove(ax)
    tp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    nudge = [0.5 / x for x in self.data.shape]
    tp[0][imax[0]] = 0 + nudge[imax[0]]
    tp[0][imax[1]] = 0 + nudge[imax[1]]
    tp[1][imax[0]] = 1 - nudge[imax[0]]
    tp[1][imax[1]] = 0 + nudge[imax[1]]
    tp[2][imax[0]] = 1 - nudge[imax[0]]
    tp[2][imax[1]] = 1 - nudge[imax[1]]
    tp[3][imax[0]] = 0 + nudge[imax[0]]
    tp[3][imax[1]] = 1 - nudge[imax[1]]
    vp[0][imax[0]] = 0
    vp[0][imax[1]] = 0
    vp[1][imax[0]] = self.data.shape[imax[0]]
    vp[1][imax[1]] = 0
    vp[2][imax[0]] = self.data.shape[imax[0]]
    vp[2][imax[1]] = self.data.shape[imax[1]]
    vp[3][imax[0]] = 0
    vp[3][imax[1]] = self.data.shape[imax[1]]
    slices = self.data.shape[ax] * self.sliceDensity
    r = list(range(slices))
    if d == -1:
        r = r[::-1]
    glBegin(GL_QUADS)
    tzVals = np.linspace(nudge[ax], 1.0 - nudge[ax], slices)
    vzVals = np.linspace(0, self.data.shape[ax], slices)
    for i in r:
        z = tzVals[i]
        w = vzVals[i]
        tp[0][ax] = z
        tp[1][ax] = z
        tp[2][ax] = z
        tp[3][ax] = z
        vp[0][ax] = w
        vp[1][ax] = w
        vp[2][ax] = w
        vp[3][ax] = w
        glTexCoord3f(*tp[0])
        glVertex3f(*vp[0])
        glTexCoord3f(*tp[1])
        glVertex3f(*vp[1])
        glTexCoord3f(*tp[2])
        glVertex3f(*vp[2])
        glTexCoord3f(*tp[3])
        glVertex3f(*vp[3])
    glEnd()