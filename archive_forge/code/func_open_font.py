import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def open_font(self, name):
    """Open the font identifed by the pattern name and return its
        font object. If name does not match any font, None is returned."""
    fid = self.display.allocate_resource_id()
    ec = error.CatchError(error.BadName)
    request.OpenFont(display=self.display, onerror=ec, fid=fid, name=name)
    self.sync()
    if ec.get_error():
        self.display.free_resource_id(fid)
        return None
    else:
        cls = self.display.get_resource_class('font', Xlib.xobject.fontable.Font)
        return cls(self.display, fid, owner=1)