from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_selection_owner(self, selection, time, onerror=None):
    request.SetSelectionOwner(display=self.display, onerror=onerror, window=self.id, selection=selection, time=time)