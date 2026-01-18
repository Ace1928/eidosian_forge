from twisted import copyright
from twisted.web import http
def handleResponsePart_with_metadata(self, data):
    self.databuffer += data
    while self.databuffer:
        stop = getattr(self, 'handle_%s' % self.metamode)()
        if stop:
            return