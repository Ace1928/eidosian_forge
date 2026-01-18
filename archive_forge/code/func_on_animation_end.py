import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
def on_animation_end(self):
    """The sprite animation reached the final frame.

            The event is triggered only if the sprite has an animation, not an
            image.  For looping animations, the event is triggered each time
            the animation loops.

            :event:
            """