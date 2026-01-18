from Xlib import X
from Xlib.protocol import rq, structs
def delete_output_property(self, output, property):
    return DeleteOutputProperty(display=self.display, opcode=self.display.get_extension_major(extname), output=output, property=property)