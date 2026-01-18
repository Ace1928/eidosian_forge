import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def freecoloroverlay(self, index):
    self.tk.call(self._w, 'freecoloroverlay', index)