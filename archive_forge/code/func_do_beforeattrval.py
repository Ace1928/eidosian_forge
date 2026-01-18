from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_beforeattrval(self, byte):
    if byte in '"\'':
        return 'attrval'
    elif byte.isspace():
        return
    elif self.beExtremelyLenient:
        if byte in lenientIdentChars or byte.isalnum():
            return 'messyattr'
        if byte == '>':
            self.attrval = 'True'
            self.tagAttributes[self.attrname] = self.attrval
            self.gotTagStart(self.tagName, self.tagAttributes)
            return self.maybeBodyData()
        if byte == '\\':
            return
    self._parseError('Invalid initial attribute value: %r; Attribute values must be quoted.' % byte)