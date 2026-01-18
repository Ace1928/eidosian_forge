from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def grab_key(self, key, modifiers, owner_events, pointer_mode, keyboard_mode, onerror=None):
    request.GrabKey(display=self.display, onerror=onerror, owner_events=owner_events, grab_window=self.id, modifiers=modifiers, key=key, pointer_mode=pointer_mode, keyboard_mode=keyboard_mode)