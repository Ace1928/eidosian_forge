from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
def check_printable(self, data):
    non_printable_match = self._get_non_printable(data)
    if non_printable_match is not None:
        start, character = non_printable_match
        position = self.index + (len(self.buffer) - self.pointer) + start
        raise ReaderError(self.name, position, ord(character), 'unicode', 'special characters are not allowed')