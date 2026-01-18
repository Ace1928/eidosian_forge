from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def change_save_set(self, mode, onerror=None):
    request.ChangeSaveSet(display=self.display, onerror=onerror, mode=mode, window=self.id)