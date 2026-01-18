import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
@property
def is_mouse_scrolling(self, *args):
    """Returns True if the touch event is a mousewheel scrolling

        .. versionadded:: 1.6.0
        """
    return 'button' in self.profile and 'scroll' in self.button