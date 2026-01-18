from Xlib import X
from Xlib.protocol import rq, structs
def input_selected(self):
    reply = InputSelected(display=self.display, opcode=self.display.get_extension_major(extname), window=self.id)
    return reply.enabled