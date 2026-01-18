import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def alloc_color_planes(self, contiguous, colors, red, green, blue):
    return request.AllocColorPlanes(display=self.display, contiguous=contiguous, cmap=self.id, colors=colors, red=red, green=green, blue=blue)