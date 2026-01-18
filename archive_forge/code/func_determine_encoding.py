from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
def determine_encoding(self):
    while not self.eof and (self.raw_buffer is None or len(self.raw_buffer) < 2):
        self.update_raw()
    if isinstance(self.raw_buffer, binary_type):
        if self.raw_buffer.startswith(codecs.BOM_UTF16_LE):
            self.raw_decode = codecs.utf_16_le_decode
            self.encoding = 'utf-16-le'
        elif self.raw_buffer.startswith(codecs.BOM_UTF16_BE):
            self.raw_decode = codecs.utf_16_be_decode
            self.encoding = 'utf-16-be'
        else:
            self.raw_decode = codecs.utf_8_decode
            self.encoding = 'utf-8'
    self.update(1)