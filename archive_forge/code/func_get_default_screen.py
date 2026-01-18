from pyglet import gl
from pyglet import app
from pyglet import window
from pyglet import canvas
def get_default_screen(self):
    """Get the default (primary) screen as specified by the user's operating system
        preferences.

        :rtype: :class:`Screen`
        """
    screens = self.get_screens()
    for screen in screens:
        if screen.x == 0 and screen.y == 0:
            return screen
    return screens[0]