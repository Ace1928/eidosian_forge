from Xlib import X
from Xlib.protocol import rq, structs
def get_rectangles(self, region):
    return GetRectangles(display=self.display, opcode=self.display.get_extension_major(extname), window=self.id, region=region)