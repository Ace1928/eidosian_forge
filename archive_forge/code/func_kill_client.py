from Xlib.protocol import request
def kill_client(self, onerror=None):
    request.KillClient(display=self.display, onerror=onerror, resource=self.id)