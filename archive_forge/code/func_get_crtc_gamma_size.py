from Xlib import X
from Xlib.protocol import rq, structs
def get_crtc_gamma_size(self, crtc):
    return GetCrtcGammaSize(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc)