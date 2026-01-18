import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def rebind_string(self, keysym, newstring):
    """Change the translation of KEYSYM to NEWSTRING.
        If NEWSTRING is None, remove old translation if any.
        """
    if newstring is None:
        try:
            del self.keysym_translations[keysym]
        except KeyError:
            pass
    else:
        self.keysym_translations[keysym] = newstring