from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def poly_fill_rectangle(self, gc, rectangles, onerror=None):
    request.PolyFillRectangle(display=self.display, onerror=onerror, drawable=self.id, gc=gc, rectangles=rectangles)