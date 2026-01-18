from .error import MarkedYAMLError
from .tokens import *
def scan_yaml_directive_value(self, start_mark):
    while self.peek() == ' ':
        self.forward()
    major = self.scan_yaml_directive_number(start_mark)
    if self.peek() != '.':
        raise ScannerError('while scanning a directive', start_mark, "expected a digit or '.', but found %r" % self.peek(), self.get_mark())
    self.forward()
    minor = self.scan_yaml_directive_number(start_mark)
    if self.peek() not in '\x00 \r\n\x85\u2028\u2029':
        raise ScannerError('while scanning a directive', start_mark, "expected a digit or ' ', but found %r" % self.peek(), self.get_mark())
    return (major, minor)