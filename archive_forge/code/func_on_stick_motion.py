import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_stick_motion(self, controller, stick, xvalue, yvalue):
    """The value of a controller analogue stick changed.

        :Parameters:
            `controller` : `Controller`
                The controller whose analogue stick changed.
            `stick` : string
                The name of the stick that changed.
            `xvalue` : float
                The current X axis value, normalized to [-1, 1].
            `yvalue` : float
                The current Y axis value, normalized to [-1, 1].
        """