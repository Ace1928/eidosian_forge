import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_pointer_mapping(self):
    """Return a list of the pointer button mappings. Entry N in the
        list sets the logical button number for the physical button N+1."""
    r = request.GetPointerMapping(display=self.display)
    return r.map