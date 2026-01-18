import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
@property
def opos(self):
    """Return the initial position of the motion event in the screen
        coordinate system (self.ox, self.oy)."""
    return (self.ox, self.oy)