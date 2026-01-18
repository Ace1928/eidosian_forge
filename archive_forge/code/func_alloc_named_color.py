import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def alloc_named_color(self, name):
    for r in rgb_res:
        m = r.match(name)
        if m:
            rs = m.group(1)
            r = string.atoi(rs + '0' * (4 - len(rs)), 16)
            gs = m.group(2)
            g = string.atoi(gs + '0' * (4 - len(gs)), 16)
            bs = m.group(3)
            b = string.atoi(bs + '0' * (4 - len(bs)), 16)
            return self.alloc_color(r, g, b)
    try:
        return request.AllocNamedColor(display=self.display, cmap=self.id, name=name)
    except error.BadName:
        return None