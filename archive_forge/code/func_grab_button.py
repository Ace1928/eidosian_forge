from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def grab_button(self, button, modifiers, owner_events, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, onerror=None):
    request.GrabButton(display=self.display, onerror=onerror, owner_events=owner_events, grab_window=self.id, event_mask=event_mask, pointer_mode=pointer_mode, keyboard_mode=keyboard_mode, confine_to=confine_to, cursor=cursor, button=button, modifiers=modifiers)