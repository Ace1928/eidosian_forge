from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def read_prev_trailer(self, xref_section_offset):
    trailer_offset = self.read_xref_table(xref_section_offset=xref_section_offset)
    m = self.re_trailer_prev.search(self.buf[trailer_offset:trailer_offset + 16384])
    check_format_condition(m, 'previous trailer not found')
    trailer_data = m.group(1)
    check_format_condition(int(m.group(2)) == xref_section_offset, "xref section offset in previous trailer doesn't match what was expected")
    trailer_dict = self.interpret_trailer(trailer_data)
    if b'Prev' in trailer_dict:
        self.read_prev_trailer(trailer_dict[b'Prev'])