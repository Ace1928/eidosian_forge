import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def alloc_color(self, red, green, blue):
    return request.AllocColor(display=self.display, cmap=self.id, red=red, green=green, blue=blue)