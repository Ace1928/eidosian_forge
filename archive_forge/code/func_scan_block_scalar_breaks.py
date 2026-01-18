from .error import MarkedYAMLError
from .tokens import *
def scan_block_scalar_breaks(self, indent):
    chunks = []
    end_mark = self.get_mark()
    while self.column < indent and self.peek() == ' ':
        self.forward()
    while self.peek() in '\r\n\x85\u2028\u2029':
        chunks.append(self.scan_line_break())
        end_mark = self.get_mark()
        while self.column < indent and self.peek() == ' ':
            self.forward()
    return (chunks, end_mark)