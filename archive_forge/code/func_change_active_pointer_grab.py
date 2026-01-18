import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def change_active_pointer_grab(self, event_mask, cursor, time, onerror=None):
    """Change the dynamic parameters of a pointer grab. See
        XChangeActivePointerGrab(3X11)."""
    request.ChangeActivePointerGrab(display=self.display, onerror=onerror, cursor=cursor, time=time, event_mask=event_mask)