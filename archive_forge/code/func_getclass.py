import sys
import glob
import inspect
def getclass(f):
    return moduleClasses(__import__(f))