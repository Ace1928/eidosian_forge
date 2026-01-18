import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_dpad_motion(self, controller, dpleft, dpright, dpup, dpdown):
    """The direction pad of the controller changed.

        :Parameters:
            `controller` : `Controller`
                The controller whose hat control changed.
            `dpleft` : boolean
                True if left is pressed on the directional pad.
            `dpright` : boolean
                True if right is pressed on the directional pad.
            `dpup` : boolean
                True if up is pressed on the directional pad.
            `dpdown` : boolean
                True if down is pressed on the directional pad.
        """