import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def freecolor(self, index):
    self.tk.call(self._w, 'freecolor', index)