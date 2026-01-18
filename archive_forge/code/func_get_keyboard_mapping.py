import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_keyboard_mapping(self, first_keycode, count):
    """Return the current keyboard mapping as a list of tuples,
        starting at first_keycount and no more than count."""
    r = request.GetKeyboardMapping(display=self.display, first_keycode=first_keycode, count=count)
    return r.keysyms