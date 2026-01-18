from Xlib import X
from Xlib.protocol import rq, structs
def query_screens(self):
    return QueryScreens(display=self.display, opcode=self.display.get_extension_major(extname))