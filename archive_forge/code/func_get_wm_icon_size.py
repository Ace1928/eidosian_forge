from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_wm_icon_size(self):
    return self._get_struct_prop(Xatom.WM_ICON_SIZE, Xatom.WM_ICON_SIZE, icccm.WMIconSize)