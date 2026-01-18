from .error import MarkedYAMLError
from .tokens import *
def scan_tag(self):
    start_mark = self.get_mark()
    ch = self.peek(1)
    if ch == '<':
        handle = None
        self.forward(2)
        suffix = self.scan_tag_uri('tag', start_mark)
        if self.peek() != '>':
            raise ScannerError('while parsing a tag', start_mark, "expected '>', but found %r" % self.peek(), self.get_mark())
        self.forward()
    elif ch in '\x00 \t\r\n\x85\u2028\u2029':
        handle = None
        suffix = '!'
        self.forward()
    else:
        length = 1
        use_handle = False
        while ch not in '\x00 \r\n\x85\u2028\u2029':
            if ch == '!':
                use_handle = True
                break
            length += 1
            ch = self.peek(length)
        handle = '!'
        if use_handle:
            handle = self.scan_tag_handle('tag', start_mark)
        else:
            handle = '!'
            self.forward()
        suffix = self.scan_tag_uri('tag', start_mark)
    ch = self.peek()
    if ch not in '\x00 \r\n\x85\u2028\u2029':
        raise ScannerError('while scanning a tag', start_mark, "expected ' ', but found %r" % ch, self.get_mark())
    value = (handle, suffix)
    end_mark = self.get_mark()
    return TagToken(value, start_mark, end_mark)