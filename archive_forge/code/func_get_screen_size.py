from Xlib import X
from Xlib.protocol import rq, structs
def get_screen_size(self, screen_no):
    """Returns the size of the given screen number"""
    return GetScreenSize(display=self.display, opcode=self.display.get_extension_major(extname), window=self.id, screen=screen_no)