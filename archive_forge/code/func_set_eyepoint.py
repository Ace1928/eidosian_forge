import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def set_eyepoint(self, distance):
    """Set how far the eye is from the position we are looking."""
    self.distance = distance
    self.tkRedraw()