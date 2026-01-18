import ssl
def feed(self, data):
    self._incoming.write(data)