from Xlib.protocol import request
from Xlib.xobject import resource, cursor
def query_text_extents(self, string):
    return request.QueryTextExtents(display=self.display, font=self.id, string=string)