import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def glDistFromLine(x, p1, p2):
    f = list(map(lambda x, y: x - y, p2, p1))
    g = list(map(lambda x, y: x - y, x, p1))
    return dot(g, g) - dot(f, g) ** 2 / dot(f, f)