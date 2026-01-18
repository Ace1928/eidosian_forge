import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_trigger_motion(self, controller, trigger, value):
    """The value of a controller analogue stick changed.

        :Parameters:
            `controller` : `Controller`
                The controller whose analogue stick changed.
            `trigger` : string
                The name of the trigger that changed.
            `value` : float
                The current value of the trigger, normalized to [-1, 1].
        """