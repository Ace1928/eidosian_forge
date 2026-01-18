import time
from paste.httpserver import WSGIServer
def serve_pending(self):
    self.reset_expires()
    while not self.stopping or self.pending:
        now = time.time()
        if now > self.expires and self.timeout:
            print('\nWARNING: WSGIRegressionServer timeout exceeded\n')
            break
        if self.pending:
            self.handle_request()
        time.sleep(0.1)