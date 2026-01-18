from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def query_tree(self):
    return request.QueryTree(display=self.display, window=self.id)