from Xlib import X
from Xlib.protocol import rq, structs
def get_output_property(self, output, property, type, longOffset, longLength):
    return GetOutputProperty(display=self.display, opcode=self.display.get_extension_major(extname), output=output, property=property, type=type, longOffset=longOffset, longLength=longLength)