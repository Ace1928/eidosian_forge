from Xlib import X
from Xlib.protocol import rq, structs
def change_output_property(self, output, property, type, format, mode, nUnits):
    return ChangeOutputProperty(display=self.display, opcode=self.display.get_extension_major(extname), output=output, property=property, type=type, format=format, mode=mode, nUnits=nUnits)