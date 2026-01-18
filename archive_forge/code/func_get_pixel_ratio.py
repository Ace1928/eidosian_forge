import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
def get_pixel_ratio(self):
    """Return the framebuffer/window size ratio.

        Some platforms and/or window systems support subpixel scaling,
        making the framebuffer size larger than the window size.
        Retina screens on OS X and Gnome on Linux are some examples.

        On a Retina systems the returned ratio would usually be 2.0 as a
        window of size 500 x 500 would have a framebuffer of 1000 x 1000.
        Fractional values between 1.0 and 2.0, as well as values above
        2.0 may also be encountered.

        :rtype: float
        :return: The framebuffer/window size ratio
        """
    return self.get_framebuffer_size()[0] / self.width