from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_attrname(self, byte):
    if byte.isalnum() or byte in identChars:
        self.attrname += byte
        return
    elif byte == '=':
        return 'beforeattrval'
    elif byte.isspace():
        return 'beforeeq'
    elif self.beExtremelyLenient:
        if byte in '"\'':
            return 'attrval'
        if byte in lenientIdentChars or byte.isalnum():
            self.attrname += byte
            return
        if byte == '/':
            self._attrname_termtag = 1
            return
        if byte == '>':
            self.attrval = 'True'
            self.tagAttributes[self.attrname] = self.attrval
            self.gotTagStart(self.tagName, self.tagAttributes)
            if self._attrname_termtag:
                self.gotTagEnd(self.tagName)
                return 'bodydata'
            return self.maybeBodyData()
        return
    self._parseError(f'Invalid attribute name: {self.attrname!r} {byte!r}')