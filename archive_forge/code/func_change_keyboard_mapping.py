import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def change_keyboard_mapping(self, first_keycode, keysyms, onerror=None):
    """Modify the keyboard mapping, starting with first_keycode.
        keysyms is a list of tuples of keysyms. keysyms[n][i] will be
        assigned to keycode first_keycode+n at index i."""
    request.ChangeKeyboardMapping(display=self.display, onerror=onerror, first_keycode=first_keycode, keysyms=keysyms)