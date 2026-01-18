import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_screen_saver(self):
    """Return an object with the attributes timeout, interval,
        prefer_blanking, allow_exposures. See XGetScreenSaver(3X11) for
        details."""
    return request.GetScreenSaver(display=self.display)