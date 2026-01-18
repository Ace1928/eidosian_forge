from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def poly_segment(self, gc, segments, onerror=None):
    request.PolySegment(display=self.display, onerror=onerror, drawable=self.id, gc=gc, segments=segments)