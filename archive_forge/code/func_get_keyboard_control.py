import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_keyboard_control(self):
    """Return an object with the following attributes:

        global_auto_repeat
            X.AutoRepeatModeOn or X.AutoRepeatModeOff.

        auto_repeats
            A list of 32 integers. List item N contains the bits for keys
            8N to 8N + 7 with the least significant bit in the byte
            representing key 8N. If a bit is on, autorepeat is enabled
            for the corresponding key.

        led_mask
            A 32-bit mask indicating which LEDs are on.

        key_click_percent
            The volume of key click, from 0 to 100.

        bell_percent

        bell_pitch

        bell_duration
            The volume, pitch and duration of the bell. """
    return request.GetKeyboardControl(display=self.display)