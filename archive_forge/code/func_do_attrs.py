from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_attrs(self, byte):
    if byte.isalnum() or byte in identChars:
        if self.tagName == '!DOCTYPE':
            return 'doctype'
        if self.tagName[0] in '!?':
            return 'waitforgt'
        return 'attrname'
    elif byte.isspace():
        return
    elif byte == '>':
        self.gotTagStart(self.tagName, self.tagAttributes)
        return not self.beExtremelyLenient and 'bodydata' or self.maybeBodyData()
    elif byte == '/':
        return 'afterslash'
    elif self.beExtremelyLenient:
        return
    self._parseError('Unexpected character: %r' % byte)