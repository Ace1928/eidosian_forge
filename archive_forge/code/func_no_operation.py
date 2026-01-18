import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def no_operation(self, onerror=None):
    """Do nothing but send a request to the server."""
    request.NoOperation(display=self.display, onerror=onerror)