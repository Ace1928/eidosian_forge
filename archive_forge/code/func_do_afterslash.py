from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_afterslash(self, byte):
    if self._after_slash_closed:
        self._parseError('Mal-formed')
    if byte != '>':
        if self.beExtremelyLenient:
            return
        else:
            self._parseError("No data allowed after '/'")
    self._after_slash_closed = 1
    self.gotTagStart(self.tagName, self.tagAttributes)
    self.gotTagEnd(self.tagName)
    return 'bodydata'