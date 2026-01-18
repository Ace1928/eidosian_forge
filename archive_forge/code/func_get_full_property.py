from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def get_full_property(self, property, type, sizehint=10):
    prop = self.get_property(property, type, 0, sizehint)
    if prop:
        val = prop.value
        if prop.bytes_after:
            prop = self.get_property(property, type, sizehint, prop.bytes_after // 4 + 1)
            val = val + prop.value
        prop.value = val
        return prop
    else:
        return None