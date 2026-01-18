import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def grab_server(self, onerror=None):
    """Disable processing of requests on all other client connections
        until the server is ungrabbed. Server grabbing should be avoided
        as much as possible."""
    request.GrabServer(display=self.display, onerror=onerror)