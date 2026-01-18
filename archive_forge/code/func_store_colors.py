import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def store_colors(self, items, onerror=None):
    request.StoreColors(display=self.display, onerror=onerror, cmap=self.id, items=items)