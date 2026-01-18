from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def put_image(self, gc, x, y, width, height, format, depth, left_pad, data, onerror=None):
    request.PutImage(display=self.display, onerror=onerror, format=format, drawable=self.id, gc=gc, width=width, height=height, dst_x=x, dst_y=y, left_pad=left_pad, depth=depth, data=data)