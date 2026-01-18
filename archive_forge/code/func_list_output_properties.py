from Xlib import X
from Xlib.protocol import rq, structs
def list_output_properties(self, output):
    return ListOutputProperties(display=self.display, opcode=self.display.get_extension_major(extname), output=output)