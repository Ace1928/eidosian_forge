from __future__ import absolute_import
import re
import sys
def append_uescape(self, char_number, escape_string):
    self._bytes.append(escape_string)
    self._unicode.append_charval(char_number)