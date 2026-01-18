from .error import MarkedYAMLError
from .tokens import *
def scan_flow_scalar_spaces(self, double, start_mark):
    chunks = []
    length = 0
    while self.peek(length) in ' \t':
        length += 1
    whitespaces = self.prefix(length)
    self.forward(length)
    ch = self.peek()
    if ch == '\x00':
        raise ScannerError('while scanning a quoted scalar', start_mark, 'found unexpected end of stream', self.get_mark())
    elif ch in '\r\n\x85\u2028\u2029':
        line_break = self.scan_line_break()
        breaks = self.scan_flow_scalar_breaks(double, start_mark)
        if line_break != '\n':
            chunks.append(line_break)
        elif not breaks:
            chunks.append(' ')
        chunks.extend(breaks)
    else:
        chunks.append(whitespaces)
    return chunks