from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
def get_snippet(self, indent=4, max_length=75):
    if self.buffer is None:
        return None
    head = ''
    start = self.pointer
    while start > 0 and self.buffer[start - 1] not in u'\x00\r\n\x85\u2028\u2029':
        start -= 1
        if self.pointer - start > max_length / 2 - 1:
            head = ' ... '
            start += 5
            break
    tail = ''
    end = self.pointer
    while end < len(self.buffer) and self.buffer[end] not in u'\x00\r\n\x85\u2028\u2029':
        end += 1
        if end - self.pointer > max_length / 2 - 1:
            tail = ' ... '
            end -= 5
            break
    snippet = utf8(self.buffer[start:end])
    caret = '^'
    caret = '^ (line: {})'.format(self.line + 1)
    return ' ' * indent + head + snippet + tail + '\n' + ' ' * (indent + self.pointer - start + len(head)) + caret