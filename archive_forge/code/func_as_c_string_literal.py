from __future__ import absolute_import
import re
import sys
def as_c_string_literal(self):
    value = split_string_literal(escape_byte_string(self))
    return '"%s"' % value