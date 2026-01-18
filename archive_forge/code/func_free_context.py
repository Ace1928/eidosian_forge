from Xlib import X
from Xlib.protocol import rq
def free_context(self, context):
    FreeContext(display=self.display, opcode=self.display.get_extension_major(extname), context=context)
    self.display.free_resource_id(context)