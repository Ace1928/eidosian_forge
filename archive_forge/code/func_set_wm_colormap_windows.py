from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_colormap_windows(self, windows, onerror=None):
    self.change_property(self.display.get_atom('WM_COLORMAP_WINDOWS'), Xatom.WINDOW, 32, [w.id for w in windows], onerror=onerror)