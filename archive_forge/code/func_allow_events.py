import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def allow_events(self, mode, time, onerror=None):
    """Release some queued events. mode should be one of
        X.AsyncPointer, X.SyncPointer, X.AsyncKeyboard, X.SyncKeyboard,
        X.ReplayPointer, X.ReplayKeyboard, X.AsyncBoth, or X.SyncBoth.
        time should be a timestamp or X.CurrentTime."""
    request.AllowEvents(display=self.display, onerror=onerror, mode=mode, time=time)