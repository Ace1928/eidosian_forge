import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def getoverlaytransparentvalue(self):
    return self.tk.getint(self.tk.call(self._w, 'getoverlaytransparentvalue'))