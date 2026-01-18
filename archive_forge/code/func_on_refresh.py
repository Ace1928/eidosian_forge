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
def on_refresh(self, dt):
    """The window contents should be redrawn.

            The `EventLoop` will dispatch this event when the `draw`
            method has been called. The window will already have the
            GL context, so there is no need to call `switch_to`. The window's
            `flip` method will be called immediately after this event, so your
            event handler should not.

            You should make no assumptions about the window contents when
            this event is triggered; a resize or expose event may have
            invalidated the framebuffer since the last time it was drawn.

            .. versionadded:: 2.0

            :event:
            """