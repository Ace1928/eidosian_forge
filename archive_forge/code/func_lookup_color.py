import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def lookup_color(self, name):
    return request.LookupColor(display=self.display, cmap=self.id, name=name)