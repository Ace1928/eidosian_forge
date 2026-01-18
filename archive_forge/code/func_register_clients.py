from Xlib import X
from Xlib.protocol import rq
def register_clients(self, context, element_header, clients, ranges):
    RegisterClients(display=self.display, opcode=self.display.get_extension_major(extname), context=context, element_header=element_header, clients=clients, ranges=ranges)