import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
@property
def ppos(self):
    """Return the previous position of the motion event in the screen
        coordinate system (self.px, self.py)."""
    return (self.px, self.py)