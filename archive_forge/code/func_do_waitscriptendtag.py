from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_waitscriptendtag(self, byte):
    self.temptagdata += byte
    if byte == '/':
        self.endtag = True
    elif not self.endtag:
        self.bodydata += '<' + self.temptagdata
        return 'waitforendscript'
    elif byte.isalnum() or byte in identChars:
        self.tagName += byte
        if not 'script'.startswith(self.tagName):
            self.bodydata += '<' + self.temptagdata
            return 'waitforendscript'
        elif self.tagName == 'script':
            self.gotText(self.bodydata)
            self.gotTagEnd(self.tagName)
            return 'waitforgt'
    elif byte.isspace():
        return 'waitscriptendtag'
    else:
        self.bodydata += '<' + self.temptagdata
        return 'waitforendscript'