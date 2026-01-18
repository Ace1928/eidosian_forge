import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
def enter_blocking(self):
    """Called by pyglet internal processes when the operating system
        is about to block due to a user interaction.  For example, this
        is common when the user begins resizing or moving a window.

        This method provides the event loop with an opportunity to set up
        an OS timer on the platform event loop, which will continue to
        be invoked during the blocking operation.

        The default implementation ensures that :py:meth:`idle` continues to be
        called as documented.

        .. versionadded:: 1.2
        """
    timeout = self.idle()
    app.platform_event_loop.set_timer(self._blocking_timer, timeout)