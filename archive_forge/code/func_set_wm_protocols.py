from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_protocols(self, protocols, onerror=None):
    self.change_property(self.display.get_atom('WM_PROTOCOLS'), Xatom.ATOM, 32, protocols, onerror=onerror)