import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def change_keyboard_control(self, onerror=None, **keys):
    """Change the parameters provided as keyword arguments:

        key_click_percent
            The volume of key clicks between 0 (off) and 100 (load).
            -1 will restore default setting.
        bell_percent
            The base volume of the bell, coded as above.
        bell_pitch
            The pitch of the bell in Hz, -1 restores the default.
        bell_duration
            The duration of the bell in milliseconds, -1 restores
            the default.
        led

        led_mode
            led_mode should be X.LedModeOff or X.LedModeOn. If led is
            provided, it should be a 32-bit mask listing the LEDs that
            should change. If led is not provided, all LEDs are changed.
        key

        auto_repeat_mode
            auto_repeat_mode should be one of X.AutoRepeatModeOff,
            X.AutoRepeatModeOn, or X.AutoRepeatModeDefault. If key is
            provided, that key will be modified, otherwise the global
            state for the entire keyboard will be modified."""
    request.ChangeKeyboardControl(display=self.display, onerror=onerror, attrs=keys)