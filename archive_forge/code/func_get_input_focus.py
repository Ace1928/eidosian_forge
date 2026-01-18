import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_input_focus(self):
    """Return an object with the following attributes:

        focus
            The window which currently holds the input
            focus, X.NONE or X.PointerRoot.
        revert_to
            Where the focus will revert, one of X.RevertToParent,
            RevertToPointerRoot, or RevertToNone. """
    return request.GetInputFocus(display=self.display)