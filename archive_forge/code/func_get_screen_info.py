from Xlib import X
from Xlib.protocol import rq, structs
def get_screen_info(self):
    """Retrieve information about the current and available configurations for
    the screen associated with this window.

    """
    return GetScreenInfo(display=self.display, opcode=self.display.get_extension_major(extname), window=self)