import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def lookup_string(self, keysym):
    """Return a string corresponding to KEYSYM, or None if no
        reasonable translation is found.
        """
    s = self.keysym_translations.get(keysym)
    if s is not None:
        return s
    import Xlib.XK
    return Xlib.XK.keysym_to_string(keysym)