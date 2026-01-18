from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def have_font(self, name):
    result = self.find_font(name)
    if result:
        if name and result.name and (result.name.lower() != name.lower()):
            return False
        return True
    else:
        return False