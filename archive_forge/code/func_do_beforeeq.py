from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_beforeeq(self, byte):
    if byte == '=':
        return 'beforeattrval'
    elif byte.isspace():
        return
    elif self.beExtremelyLenient:
        if byte.isalnum() or byte in identChars:
            self.attrval = 'True'
            self.tagAttributes[self.attrname] = self.attrval
            return 'attrname'
        elif byte == '>':
            self.attrval = 'True'
            self.tagAttributes[self.attrname] = self.attrval
            self.gotTagStart(self.tagName, self.tagAttributes)
            if self._beforeeq_termtag:
                self.gotTagEnd(self.tagName)
                return 'bodydata'
            return self.maybeBodyData()
        elif byte == '/':
            self._beforeeq_termtag = 1
            return
    self._parseError('Invalid attribute')