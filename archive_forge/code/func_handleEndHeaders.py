from twisted import copyright
from twisted.web import http
def handleEndHeaders(self):
    if self.got_metadata:
        self.handleResponsePart = self.handleResponsePart_with_metadata
    else:
        self.handleResponsePart = self.gotMP3Data