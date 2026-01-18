from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def set_wm_state(self, hints={}, onerror=None, **keys):
    atom = self.display.get_atom('WM_STATE')
    self._set_struct_prop(atom, atom, icccm.WMState, hints, keys, onerror)