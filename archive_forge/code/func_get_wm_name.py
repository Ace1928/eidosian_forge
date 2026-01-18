from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_name(self):
    d = self.get_full_property(Xatom.WM_NAME, Xatom.STRING)
    if d is None or d.format != 8:
        return None
    else:
        return d.value