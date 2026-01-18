import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_joyaxis_motion(self, joystick, axis, value):
    """The value of a joystick axis changed.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose axis changed.
            `axis` : string
                The name of the axis that changed.
            `value` : float
                The current value of the axis, normalized to [-1, 1].
        """