from .error import MarkedYAMLError
from .tokens import *
def scan_block_scalar_indentation(self):
    chunks = []
    max_indent = 0
    end_mark = self.get_mark()
    while self.peek() in ' \r\n\x85\u2028\u2029':
        if self.peek() != ' ':
            chunks.append(self.scan_line_break())
            end_mark = self.get_mark()
        else:
            self.forward()
            if self.column > max_indent:
                max_indent = self.column
    return (chunks, max_indent, end_mark)