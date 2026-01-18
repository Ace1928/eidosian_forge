import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def set_screen_saver(self, timeout, interval, prefer_blank, allow_exposures, onerror=None):
    """See XSetScreenSaver(3X11)."""
    request.SetScreenSaver(display=self.display, onerror=onerror, timeout=timeout, interval=interval, prefer_blank=prefer_blank, allow_exposures=allow_exposures)