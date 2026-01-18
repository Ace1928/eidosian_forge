from .error import MarkedYAMLError
from .tokens import *
def scan_tag_uri(self, name, start_mark):
    chunks = []
    length = 0
    ch = self.peek(length)
    while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or (ch in "-;/?:@&=+$,_.!~*'()[]%"):
        if ch == '%':
            chunks.append(self.prefix(length))
            self.forward(length)
            length = 0
            chunks.append(self.scan_uri_escapes(name, start_mark))
        else:
            length += 1
        ch = self.peek(length)
    if length:
        chunks.append(self.prefix(length))
        self.forward(length)
        length = 0
    if not chunks:
        raise ScannerError('while parsing a %s' % name, start_mark, 'expected URI, but found %r' % ch, self.get_mark())
    return ''.join(chunks)