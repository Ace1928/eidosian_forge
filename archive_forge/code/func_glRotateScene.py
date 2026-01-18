import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def glRotateScene(s, xcenter, ycenter, zcenter, x, y, mousex, mousey):
    glMatrixMode(GL_MODELVIEW)
    mat = glGetDoublev(GL_MODELVIEW_MATRIX)
    glLoadIdentity()
    glTranslatef(xcenter, ycenter, zcenter)
    glRotatef(s * (y - mousey), 1.0, 0.0, 0.0)
    glRotatef(s * (x - mousex), 0.0, 1.0, 0.0)
    glTranslatef(-xcenter, -ycenter, -zcenter)
    glMultMatrixd(mat)