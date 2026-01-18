import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_button_release(self, button):
    """A button on the remote was released.

        The 'select_hold' and 'menu_hold' button release events are sent
        immediately after the corresponding press events regardless of
        whether the user has released the button.

        :Parameters:
            `button` : str
                The name of the button that was released. The valid names are
                'up', 'down', 'left', 'right', 'left_hold', 'right_hold',
                'menu', 'menu_hold', 'select', and 'select_hold'

        :event:
        """