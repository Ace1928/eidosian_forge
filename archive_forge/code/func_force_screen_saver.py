import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def force_screen_saver(self, mode, onerror=None):
    """If mode is X.ScreenSaverActive the screen saver is activated.
        If it is X.ScreenSaverReset, the screen saver is deactivated as
        if device input had been received."""
    request.ForceScreenSaver(display=self.display, onerror=onerror, mode=mode)