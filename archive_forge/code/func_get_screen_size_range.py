from Xlib import X
from Xlib.protocol import rq, structs
def get_screen_size_range(self):
    """Retrieve the range of possible screen sizes. The screen may be set to
	any size within this range.

    """
    return GetScreenSizeRange(display=self.display, opcode=self.display.get_extension_major(extname), window=self)