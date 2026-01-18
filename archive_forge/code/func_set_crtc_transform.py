from Xlib import X
from Xlib.protocol import rq, structs
def set_crtc_transform(self, crtc, n_bytes_filter):
    return SetCrtcTransform(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc, n_bytes_filter=n_bytes_filter)