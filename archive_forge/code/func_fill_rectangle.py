from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def fill_rectangle(self, gc, x, y, width, height, onerror=None):
    request.PolyFillRectangle(display=self.display, onerror=onerror, drawable=self.id, gc=gc, rectangles=[(x, y, width, height)])