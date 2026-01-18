import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def basic_lighting(self):
    """        Set up some basic lighting (single infinite light source).

        Also switch on the depth buffer."""
    self.activate()
    light_position = (1, 1, 1, 0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()