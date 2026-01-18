from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def poly_line(self, gc, coord_mode, points, onerror=None):
    request.PolyLine(display=self.display, onerror=onerror, coord_mode=coord_mode, drawable=self.id, gc=gc, points=points)