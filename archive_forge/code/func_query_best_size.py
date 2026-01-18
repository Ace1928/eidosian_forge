from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def query_best_size(self, item_class, width, height):
    return request.QueryBestSize(display=self.display, item_class=item_class, drawable=self.id, width=width, height=height)