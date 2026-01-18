from .error import MarkedYAMLError
from .tokens import *
def scan_flow_scalar_breaks(self, double, start_mark):
    chunks = []
    while True:
        prefix = self.prefix(3)
        if (prefix == '---' or prefix == '...') and self.peek(3) in '\x00 \t\r\n\x85\u2028\u2029':
            raise ScannerError('while scanning a quoted scalar', start_mark, 'found unexpected document separator', self.get_mark())
        while self.peek() in ' \t':
            self.forward()
        if self.peek() in '\r\n\x85\u2028\u2029':
            chunks.append(self.scan_line_break())
        else:
            return chunks