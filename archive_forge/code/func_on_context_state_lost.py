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
def on_context_state_lost(self):
    """The state of the window's GL context was lost.

            pyglet may sometimes need to recreate the window's GL context if
            the window is moved to another video device, or between fullscreen
            or windowed mode.  In this case it will try to share the objects
            (display lists, texture objects, shaders) between the old and new
            contexts.  If this is possible, only the current state of the GL
            context is lost, and the application should simply restore state.

            :event:
            """