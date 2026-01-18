from Xlib import X
from Xlib.protocol import rq
def grab_control(self, impervious):
    GrabControl(display=self.display, opcode=self.display.get_extension_major(extname), impervious=impervious)