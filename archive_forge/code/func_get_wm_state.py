from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_state(self):
    atom = self.display.get_atom('WM_STATE')
    return self._get_struct_prop(atom, atom, icccm.WMState)