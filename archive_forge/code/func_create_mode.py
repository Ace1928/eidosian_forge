from Xlib import X
from Xlib.protocol import rq, structs
def create_mode(self):
    return CreateMode(display=self.display, opcode=self.display.get_extension_major(extname), window=self)