from twisted import copyright
from twisted.web import http
def handle_length(self):
    self.remaining = ord(self.databuffer[0]) * 16
    self.databuffer = self.databuffer[1:]
    self.metamode = 'meta'