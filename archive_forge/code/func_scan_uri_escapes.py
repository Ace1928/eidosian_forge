from .error import MarkedYAMLError
from .tokens import *
def scan_uri_escapes(self, name, start_mark):
    codes = []
    mark = self.get_mark()
    while self.peek() == '%':
        self.forward()
        for k in range(2):
            if self.peek(k) not in '0123456789ABCDEFabcdef':
                raise ScannerError('while scanning a %s' % name, start_mark, 'expected URI escape sequence of 2 hexdecimal numbers, but found %r' % self.peek(k), self.get_mark())
        codes.append(int(self.prefix(2), 16))
        self.forward(2)
    try:
        value = bytes(codes).decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ScannerError('while scanning a %s' % name, start_mark, str(exc), mark)
    return value