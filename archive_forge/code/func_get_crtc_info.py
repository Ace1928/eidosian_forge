from Xlib import X
from Xlib.protocol import rq, structs
def get_crtc_info(self, crtc, config_timestamp):
    return GetCrtcInfo(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc, config_timestamp=config_timestamp)