import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def query_colors(self, pixels):
    r = request.QueryColors(display=self.display, cmap=self.id, pixels=pixels)
    return r.colors