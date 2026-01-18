from Xlib import X
from Xlib.protocol import rq
def compare_cursor(self, cursor):
    r = CompareCursor(display=self.display, opcode=self.display.get_extension_major(extname), window=self.id, cursor=cursor)
    return r.same