import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def free_colors(self, pixels, plane_mask, onerror=None):
    request.FreeColors(display=self.display, onerror=onerror, cmap=self.id, plane_mask=plane_mask, pixels=pixels)