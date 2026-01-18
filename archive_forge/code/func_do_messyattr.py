from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_messyattr(self, byte):
    if byte.isspace():
        return 'attrs'
    elif byte == '>':
        endTag = 0
        if self.attrval.endswith('/'):
            endTag = 1
            self.attrval = self.attrval[:-1]
        self.tagAttributes[self.attrname] = self.attrval
        self.gotTagStart(self.tagName, self.tagAttributes)
        if endTag:
            self.gotTagEnd(self.tagName)
            return 'bodydata'
        return self.maybeBodyData()
    else:
        self.attrval += byte