from twisted import copyright
from twisted.web import http
def handle_meta(self):
    if len(self.databuffer) >= self.remaining:
        if self.remaining:
            data = self.databuffer[:self.remaining]
            self.gotMetaData(self.parseMetadata(data))
        self.databuffer = self.databuffer[self.remaining:]
        self.metamode = 'mp3'
    else:
        return 1