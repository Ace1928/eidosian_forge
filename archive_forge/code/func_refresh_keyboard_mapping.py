import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def refresh_keyboard_mapping(self, evt):
    """This method should be called once when a MappingNotify event
        is received, to update the keymap cache. evt should be the event
        object."""
    if isinstance(evt, event.MappingNotify):
        if evt.request == X.MappingKeyboard:
            self._update_keymap(evt.first_keycode, evt.count)
    else:
        raise TypeError('expected a MappingNotify event')