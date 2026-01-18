from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def fill_arc(self, gc, x, y, width, height, angle1, angle2, onerror=None):
    request.PolyFillArc(display=self.display, onerror=onerror, drawable=self.id, gc=gc, arcs=[(x, y, width, height, angle1, angle2)])