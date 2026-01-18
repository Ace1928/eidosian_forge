import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def alloccoloroverlay(self, red, green, blue):
    return self.tk.getint(self.tk.call(self._w, 'alloccoloroverlay', red, green, blue))