import re
import docutils
from docutils import nodes, writers, languages
def append_header(self):
    """append header with .TH and .SH NAME"""
    if self.header_written:
        return
    self.head.append(self.header())
    self.head.append(MACRO_DEF)
    self.header_written = 1