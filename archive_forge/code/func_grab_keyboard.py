from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def grab_keyboard(self, owner_events, pointer_mode, keyboard_mode, time):
    r = request.GrabKeyboard(display=self.display, owner_events=owner_events, grab_window=self.id, time=time, pointer_mode=pointer_mode, keyboard_mode=keyboard_mode)
    return r.status