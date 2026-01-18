import py
import sys
from inspect import CO_VARARGS, CO_VARKEYWORDS, isclass
import traceback
def getlocals(self):
    return self.frame.f_locals