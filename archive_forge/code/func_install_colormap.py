import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def install_colormap(self, onerror=None):
    request.InstallColormap(display=self.display, onerror=onerror, cmap=self.id)