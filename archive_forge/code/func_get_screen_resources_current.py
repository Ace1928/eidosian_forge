from Xlib import X
from Xlib.protocol import rq, structs
def get_screen_resources_current(self):
    return GetScreenResourcesCurrent(display=self.display, opcode=self.display.get_extension_major(extname), window=self)