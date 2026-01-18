from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_class(self, inst, cls, onerror=None):
    self.change_property(Xatom.WM_CLASS, Xatom.STRING, 8, '%s\x00%s\x00' % (inst, cls), onerror=onerror)