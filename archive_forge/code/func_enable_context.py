from Xlib import X
from Xlib.protocol import rq
def enable_context(self, context, callback):
    EnableContext(callback=callback, display=self.display, opcode=self.display.get_extension_major(extname), context=context)