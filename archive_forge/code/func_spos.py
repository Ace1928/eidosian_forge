import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
@property
def spos(self):
    """Return the position in the 0-1 coordinate system (self.sx, self.sy).
        """
    return (self.sx, self.sy)