import ctypes, logging, os, sys
from ctypes import util
import OpenGL
def buildFunction(functionType, name, dll):
    """Abstract away the ctypes function-creation operation"""
    return functionType((name, dll))