from .error import MarkedYAMLError
from .tokens import *
def scan_plain(self):
    chunks = []
    start_mark = self.get_mark()
    end_mark = start_mark
    indent = self.indent + 1
    spaces = []
    while True:
        length = 0
        if self.peek() == '#':
            break
        while True:
            ch = self.peek(length)
            if ch in '\x00 \t\r\n\x85\u2028\u2029' or (ch == ':' and self.peek(length + 1) in '\x00 \t\r\n\x85\u2028\u2029' + (u',[]{}' if self.flow_level else u'')) or (self.flow_level and ch in ',?[]{}'):
                break
            length += 1
        if length == 0:
            break
        self.allow_simple_key = False
        chunks.extend(spaces)
        chunks.append(self.prefix(length))
        self.forward(length)
        end_mark = self.get_mark()
        spaces = self.scan_plain_spaces(indent, start_mark)
        if not spaces or self.peek() == '#' or (not self.flow_level and self.column < indent):
            break
    return ScalarToken(''.join(chunks), True, start_mark, end_mark)