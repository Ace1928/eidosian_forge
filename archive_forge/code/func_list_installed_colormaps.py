from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def list_installed_colormaps(self):
    r = request.ListInstalledColormaps(display=self.display, window=self.id)
    return r.cmaps