from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
def forward_1_1(self, length=1):
    if self.pointer + length + 1 >= len(self.buffer):
        self.update(length + 1)
    while length != 0:
        ch = self.buffer[self.pointer]
        self.pointer += 1
        self.index += 1
        if ch in u'\n\x85\u2028\u2029' or (ch == u'\r' and self.buffer[self.pointer] != u'\n'):
            self.line += 1
            self.column = 0
        elif ch != u'\ufeff':
            self.column += 1
        length -= 1