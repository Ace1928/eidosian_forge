from Xlib import X
from Xlib.protocol import rq, structs
def set_output_primary(self, output):
    return SetOutputPrimary(display=self.display, opcode=self.display.get_extension_major(extname), window=self, output=output)