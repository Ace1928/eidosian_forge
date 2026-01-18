from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_colormap_windows(self):
    d = self.get_full_property(self.display.get_atom('WM_COLORMAP_WINDOWS'), Xatom.WINDOW)
    if d is None or d.format != 32:
        return []
    else:
        cls = self.display.get_resource_class('window', Window)
        return list(map(lambda i, d=self.display, c=cls: c(d, i), d.value))