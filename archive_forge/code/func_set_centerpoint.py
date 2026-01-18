import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def set_centerpoint(self, x, y, z):
    """Set the new center point for the model.
        This is where we are looking."""
    self.xcenter = x
    self.ycenter = y
    self.zcenter = z
    self.tkRedraw()