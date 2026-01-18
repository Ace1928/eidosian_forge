from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def change_attributes(self, onerror=None, **keys):
    request.ChangeWindowAttributes(display=self.display, onerror=onerror, window=self.id, attrs=keys)