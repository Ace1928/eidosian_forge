from Xlib import X
from Xlib.protocol import rq, structs
def destroy_mode(self, mode):
    return DestroyMode(display=self.display, opcode=self.display.get_extension_major(extname), mode=mode)