from Xlib import X
from Xlib.protocol import rq, structs
def get_output_primary(self):
    return GetOutputPrimary(display=self.display, opcode=self.display.get_extension_major(extname), window=self)