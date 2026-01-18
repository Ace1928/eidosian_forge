from Xlib import X
from Xlib.protocol import rq, structs
def set_screen_size(self, width, height, width_in_millimeters=None, height_in_millimeters=None):
    return SetScreenSize(display=self.display, opcode=self.display.get_extension_major(extname), window=self, width=width, height=height, width_in_millimeters=width_in_millimeters, height_in_millimeters=height_in_millimeters)