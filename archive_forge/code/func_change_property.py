from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def change_property(self, property, type, format, data, mode=X.PropModeReplace, onerror=None):
    request.ChangeProperty(display=self.display, onerror=onerror, mode=mode, window=self.id, property=property, type=type, data=(format, data))