from Xlib import X
from Xlib.protocol import rq, structs
def get_crtc_transform(self, crtc):
    return GetCrtcTransform(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc)