import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_atom_name(self, atom):
    """Look up the name of atom, returning it as a string. Will raise
        BadAtom if atom does not exist."""
    r = request.GetAtomName(display=self.display, atom=atom)
    return r.name