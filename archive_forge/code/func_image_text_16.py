from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def image_text_16(self, gc, x, y, string, onerror=None):
    request.ImageText16(display=self.display, onerror=onerror, drawable=self.id, gc=gc, x=x, y=y, string=string)